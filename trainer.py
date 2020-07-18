import os
import torch
from utils import make_noise
from torch_tools.visualization import to_image
from matplotlib import pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
from visualization import fig_to_image
from torch_tools.visualization import to_image

from torchvision import transforms
from latent_deformator import DeformatorType, normal_projection_stat
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
        self.shift_scale = 6.0
        self.min_shift = 0.5
        self.shift_distribution = ShiftDistribution.UNIFORM

        self.deformator_lr = 0.0001
        self.predictor_lr = 0.0001

        self.label_weight = 1.0
        self.shift_weight = 0.25
        self.deformation_loss_weight = 2.0

        self.n_steps = 50000 + 1
        self.batch_size = 12

        self.z_norm_loss_low_bound = 1.1
        self.z_mean_weight = 200.0
        self.z_std_weight = 200.0

        self.steps_per_log = 50
        self.steps_per_save = 2000
        self.steps_per_img_log = 2000
        self.steps_per_backup = 2000

        self.max_latent_ind = 128
        self.efros_threshold = 0.1

        for key, val in kwargs.items():
            if val is not None:
                self.__dict__[key] = val


class Trainer(object):
    def __init__(self, params=Params(), out_dir='', verbose=False):
        if verbose:
            print('Trainer inited with:\n{}'.format(str(params.__dict__)))
        self.p = params
        self.out_dir = out_dir
        self.models_dir = os.path.join(out_dir, 'models')
        self.images_dir = os.path.join(out_dir, 'images')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.checkpoint = os.path.join(out_dir, 'checkpoint.pt')

    def log(self, step, logit_loss, shift_loss, z_loss, loss):
            print(f'Step {step} | {logit_loss:.3} | {shift_loss:.3} | {z_loss.item():.3} | Loss: {loss.item():.3}')

    def make_shifts(self, latent_dim, target_indices=None):
        if target_indices is None:
            target_indices = torch.randint(0, self.p.max_latent_ind, [self.p.batch_size], device='cuda')
        if self.p.shift_distribution == ShiftDistribution.NORMAL:
            shifts =  torch.randn(target_indices.shape, device='cuda')
        elif self.p.shift_distribution == ShiftDistribution.UNIFORM:
            shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

        shifts = self.p.shift_scale * shifts
        shifts[(shifts < self.p.min_shift) & (shifts > 0)] = self.p.min_shift
        shifts[(shifts > -self.p.min_shift) & (shifts < 0)] = -self.p.min_shift

        if isinstance(latent_dim, int):
            latent_dim = [latent_dim]
        z_shift = torch.zeros([self.p.batch_size] + latent_dim, device='cuda')
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            #z_shift[i][4*index:4*(index+1)] += val
            z_shift[i][index] += val

        return target_indices, shifts, z_shift

    # def make_shifts(self, latent_dim, target_indices=None):
    #     if target_indices is None:
    #         target_indices = torch.randint(0, 2, [self.p.batch_size], device='cuda')
    #
    #     shifts = torch.randn(target_indices.shape, device='cuda')
    #     shifts = self.p.shift_scale * shifts
    #     shifts[(shifts < self.p.min_shift) & (shifts > 0)] = self.p.min_shift
    #     shifts[(shifts > -self.p.min_shift) & (shifts < 0)] = -self.p.min_shift
    #
    #     z = torch.randn((len(target_indices), latent_dim), device='cuda')
    #     for i in range(self.p.batch_size):
    #         if target_indices[i] == 0:
    #             z[i, :latent_dim // 2] += shifts[i]
    #         else:
    #             z[i, latent_dim // 2:] += shifts[i]
    #     return target_indices, shifts, z

    def start_from_checkpoint(self, deformator, predictor):
        step = 0
        if os.path.isfile(self.checkpoint):
            state_dict = torch.load(self.checkpoint)
            step = state_dict['step']
            deformator.load_state_dict(state_dict['deformator'])
            predictor.load_state_dict(state_dict['predictor'])
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, deformator, predictor, step):
        state_dict = {
            'step': step,
            'deformator': deformator.state_dict(),
            'predictor': predictor.state_dict(),
        }
        torch.save(state_dict, self.checkpoint)

    def save_models(self, deformator, predictor, step):
        torch.save(deformator.state_dict(),
                   os.path.join(self.models_dir, 'deformator_{}.pt'.format(step)))
        torch.save(predictor.state_dict(),
                   os.path.join(self.models_dir, 'predictor_{}.pt'.format(step)))

    def train(self, G, deformator, predictor, efros_model, inception=None):
        G.cuda().eval()
        efros_model.cuda().eval()
        deformator.cuda().train()
        predictor.cuda().train()

        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].cuda()
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].cuda()
        normalize = lambda x: (x - mean) / std

        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=self.p.deformator_lr) \
            if deformator.type not in [DeformatorType.ID, DeformatorType.RANDOM] else None
        shift_predictor_opt = torch.optim.Adam(predictor.parameters(), lr=self.p.predictor_lr)

        recovered_step = self.start_from_checkpoint(deformator, predictor)
        for step in range(recovered_step, self.p.n_steps, 1):
            deformator.zero_grad()
            predictor.zero_grad()

            with torch.no_grad():
                z = make_noise(self.p.batch_size, G.dim_z).cuda()
                for _ in range(5):
                    imgs = G([z])[0].clamp(-1, 1)
                    normalized_imgs = normalize(F.interpolate(0.5 * (imgs + 1), predictor.downsample))
                    scores = torch.sigmoid(efros_model(normalized_imgs).view(-1))
                    if (scores < self.p.efros_threshold).all():
                        break
                    z[scores > self.p.efros_threshold] = make_noise(len(z[scores > self.p.efros_threshold]), G.dim_z).cuda()

            z_orig = torch.clone(z)
            target_indices, shifts, z_shift = self.make_shifts(G.dim_z)

            # Deformation
            if self.p.global_deformation:
                z_shifted = deformator(z + z_shift)
                z = deformator(z)
            else:
                z_shifted = z + deformator(z_shift)

            imgs_shifted = G([z_shifted])[0].clamp(-1, 1)
            logits, shift_predictions = predictor(imgs, imgs_shifted)
            logit_loss = self.p.label_weight * self.cross_entropy(logits, target_indices)
            shift_loss = self.p.shift_weight * torch.mean(torch.abs(shift_predictions - shifts))

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

            if step % self.p.steps_per_log == 0:
                self.log(step, logit_loss, shift_loss, z_loss, loss)

            if step % self.p.steps_per_save == 0:
                self.save_checkpoint(deformator, predictor, step)

            if step % self.p.steps_per_img_log == 0:
                z = make_noise(self.p.max_latent_ind, G.dim_z).cuda()
                z_shifted = torch.zeros([self.p.max_latent_ind] + [G.dim_z], device='cuda')
                for i in range(self.p.max_latent_ind):
                    z_shift = torch.zeros(G.dim_z, device='cuda')
                    z_shift[i] += self.p.shift_scale
                    z_shifted[i] = z[i] + z_shift

                fig, axes = plt.subplots(self.p.max_latent_ind, 8, figsize=(20, 10 * self.p.max_latent_ind // 4))
                for dir_id in range(0, self.p.max_latent_ind, 4):
                    for i in range(4):
                        with torch.no_grad():
                            img = G([z[dir_id + i][None]])[0]
                            img_shifted = G([z_shifted[dir_id + i][None]])[0]

                        img = to_image(img.cpu().clamp(-1, 1))
                        axes[dir_id // 4, 2*i].imshow(img)
                        axes[dir_id // 4, 2*i].axis('off')
                        axes[dir_id // 4, 2*i].set_title(f"Orig | Dim {dir_id}")

                        img_shifted = to_image(img_shifted.cpu().clamp(-1, 1))
                        axes[dir_id // 4, 2*i + 1].imshow(img_shifted)
                        axes[dir_id // 4, 2*i + 1].axis('off')
                        axes[dir_id // 4, 2*i + 1].set_title(f"Shifted | Dim {dir_id}")

                fig_to_image(fig).save(os.path.join(self.out_dir, f"step{step}.png"))
                plt.close(fig)

