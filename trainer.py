import os
import torch
from utils import make_noise
from torch_tools.visualization import to_image
from matplotlib import pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
from visualization import fig_to_image

from latent_deformator import DeformatorType, normal_projection_stat
from latent_shift_predictor import ResNetShiftPredictor
from enum import Enum


class DeformatorLoss(Enum):
    L2 = 0,
    RELATIVE = 1,
    STAT = 2,
    NONE = 3,


class ShiftDistribution(Enum):
    NORMAL = 0,
    UNIFORM = 1,


class Params(object):
    def __init__(self, **kwargs):
        self.global_deformation = False
        self.deformation_loss = DeformatorLoss.NONE
        self.shift_scale = 8.0  # 6.0
        self.min_shift = 0.5
        self.shift_distribution = ShiftDistribution.UNIFORM

        self.deformator_lr = 0.0001
        self.shift_predictor_lr = 0.0001

        self.label_weight = 2.0
        self.shift_weight = 0.5
        self.deformation_loss_weight = 2.0

        self.n_steps = int(1e+5) + 1
        self.batch_size = 32

        self.z_norm_loss_low_bound = 1.1
        self.z_mean_weight = 200.0
        self.z_std_weight = 200.0

        self.l2_loss_weight = 10.0

        self.steps_per_log = 100
        self.steps_per_save = 10000
        self.steps_per_img_log = 1000
        self.steps_per_backup = 1000

        self.max_latent_ind = 512

        for key, val in kwargs.items():
            if val is not None:
                self.__dict__[key] = val


class Trainer(object):
    def __init__(self, params=Params(), out_dir='', verbose=False):
        if verbose:
            print('Trainer inited with:\n{}'.format(str(params.__dict__)))
        self.p = params
        self.log_dir = out_dir
        self.models_dir = os.path.join(out_dir, 'models')
        self.images_dir = os.path.join(out_dir, 'images')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def log(self, step, loss):
            print('Step {} loss: {:.3}'.format(step, loss.item()))

    def make_shifts(self, latent_dim, target_indices=None):
        if target_indices is None:
            target_indices = torch.randint(0, self.p.max_latent_ind, [self.p.batch_size], device='cuda')
        if self.p.shift_distribution == ShiftDistribution.NORMAL:
            shifts = torch.randn(target_indices.shape, device='cuda')
        elif self.p.shift_distribution == ShiftDistribution.UNIFORM:
            shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

        shifts = self.p.shift_scale * shifts
        shifts[(shifts < self.p.min_shift) & (shifts > 0)] = self.p.min_shift
        shifts[(shifts > -self.p.min_shift) & (shifts < 0)] = -self.p.min_shift

        if isinstance(latent_dim, int):
            latent_dim = [latent_dim]
        z_shift = torch.zeros([self.p.batch_size] + latent_dim, device='cuda')
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift

    def start_from_checkpoint(self, deformator, shift_predictor):
        step = 0
        if os.path.isfile(self.checkpoint):
            state_dict = torch.load(self.checkpoint)
            step = state_dict['step']
            deformator.load_state_dict(state_dict['deformator'])
            shift_predictor.load_state_dict(state_dict['shift_predictor'])
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, deformator, shift_predictor, step):
        state_dict = {
            'step': step,
            'deformator': deformator.state_dict(),
            'shift_predictor': shift_predictor.state_dict(),
        }
        torch.save(state_dict, self.checkpoint)

    def save_models(self, deformator, shift_predictor, step):
        torch.save(deformator.state_dict(),
                   os.path.join(self.models_dir, 'deformator_{}.pt'.format(step)))
        torch.save(shift_predictor.state_dict(),
                   os.path.join(self.models_dir, 'shift_predictor_{}.pt'.format(step)))

    def train(self, G, deformator, shift_predictor, inception=None):
        G.cuda().eval()
        deformator.cuda().train()
        shift_predictor.cuda().train()

        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=self.p.deformator_lr) \
            if deformator.type not in [DeformatorType.ID, DeformatorType.RANDOM] else None
        shift_predictor_opt = torch.optim.Adam(
            shift_predictor.parameters(), lr=self.p.shift_predictor_lr)

        recovered_step = self.start_from_checkpoint(deformator, shift_predictor)
        for step in range(recovered_step, self.p.n_steps, 1):
            G.zero_grad()
            deformator.zero_grad()
            shift_predictor.zero_grad()

            z = make_noise(self.p.batch_size, G.style_dim).cuda()
            z_orig = torch.clone(z)
            target_indices, shifts, z_shift = self.make_shifts(G.dim_z)

            # Deformation

            if self.p.global_deformation:
                z_shifted = deformator(z + z_shift)
                z = deformator(z)
            else:
                z_shifted = z + deformator(z_shift)

            imgs = G(z)
            imgs_shifted = G(z_shifted)

            logits, shift_prediction = shift_predictor(imgs, imgs_shifted)
            logit_loss = self.p.label_weight * self.cross_entropy(logits, target_indices)
            shift_loss = self.p.shift_weight * torch.mean(torch.abs(shift_prediction - shifts))

            # Loss

            # deformator penalty
            if self.p.deformation_loss == DeformatorLoss.STAT:
                z_std, z_mean = normal_projection_stat(z)
                z_loss = self.p.z_mean_weight * torch.abs(z_mean) + \
                         self.p.z_std_weight * torch.abs(1.0 - z_std)

            elif self.p.deformation_loss == DeformatorLoss.L2:
                z_loss = self.p.deformation_loss_weight * torch.mean(torch.norm(z, dim=1))
                if z_loss < self.p.z_norm_loss_low_bound * torch.mean(torch.norm(z_orig, dim=1)):
                    z_loss = torch.tensor([0.0], device='cuda')

            elif self.p.deformation_loss == DeformatorLoss.RELATIVE:
                deformation_norm = torch.norm(z - z_shifted, dim=1)
                z_loss = self.p.deformation_loss_weight * torch.mean(torch.abs(deformation_norm - shifts))

            else:
                z_loss = torch.tensor([0.0], device='cuda')

            # total loss
            loss = logit_loss + shift_loss + z_loss
            loss.backward()

            if deformator_opt is not None:
                deformator_opt.step()
            shift_predictor_opt.step()

            self.log(step, loss)
