import os
import torch
from utils import make_noise
from torch_tools.visualization import to_image
from matplotlib import pyplot as plt

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
from visualization import fig_to_image


class Params(object):
    def __init__(self, **kwargs):
        self.n_steps = int(1e+5) + 1
        self.batch_size = 32

        self.z_norm_loss_low_bound = 1.1
        self.z_mean_weight = 200.0
        self.z_std_weight = 200.0

        self.steps_per_log = 100
        self.steps_per_save = 10000

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

    def train(self, G, inception):
        # transform = Compose([
        #     Resize(299),
        #     ToTensor(),
        #     Normalize(mean=[0.485, 0.456, 0.406],
        #               std=[0.229, 0.224, 0.225])
        # ])

        target_feats = torch.tensor(np.load("stats/imagenet_gaussian_mean.npy"))[None]
        self.p.batch_size = 1
        # target_feats = torch.tensor(np.load("stats/imagenet_gaussian_directions.npy"))
        G.cuda().eval()

        z_orig = make_noise(self.p.batch_size, G.dim_z).cuda()
        z_adv = nn.Parameter(z_orig, requires_grad=True)
        optimizer = torch.optim.Adam([z_adv], lr=0.003, betas=(0.9, 0.999))


        os.makedirs("inv_samples", exist_ok=True)
        torch.save(z_orig, "inv_samples/orig_z.pt")

        for step in range(0, self.p.n_steps, 1):
            G.zero_grad()
            optimizer.zero_grad()

            imgs_adv = G(z_adv)
            imgs_adv = ((imgs_adv + 1.) / 2.).clamp(0, 1)
            imgs_adv = F.interpolate(imgs_adv, size=(299, 299),
                                     mode='bilinear', align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
            std = torch.tensor([0.229, 0.224, 0.225]).cuda()
            imgs_adv = (imgs_adv - mean[..., None, None]) / std[..., None, None]

            img_adv_feats = inception(imgs_adv)
            loss = ((target_feats - img_adv_feats) ** 2).mean()
            loss.backward()
            optimizer.step()

            if step % self.p.steps_per_log == 0:
                self.log(step, loss)
            if step % self.p.steps_per_save == 0:
                torch.save(z_adv.data, f"inv_samples/mean_inv_z_{step}.pt")
                fig, axes = plt.subplots(1, self.p.batch_size, figsize=(12, 12))
                for i in range(len(imgs_adv)):
                    axes[i].imshow(to_image(imgs_adv[i]))
                    axes[i].set_title(f"Inversion {i}")

                fig_to_image(fig).save(f"inv_samples/mean_step{step}.png")
                plt.close(fig)




