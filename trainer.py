import os
import torch
from utils import make_noise
from torch_tools.visualization import to_image
from matplotlib import pyplot as plt

from PIL import Image
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
        self.steps_per_save = 5000

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
        #     # Normalize(mean=[0.485, 0.456, 0.406],
        #     #           std=[0.229, 0.224, 0.225])
        # ])
        # target_img = transform(Image.open("../datasets/imagenet_crop128/val/239/0.png")).cuda()[None]
        # with torch.no_grad():
        #     target_feats = inception(2*target_img - 1)

        print('Find the nearest sample')
        # for class_idx in range(0, 1000, 5):
        # target_feats = torch.tensor(np.load("stats/imagenet_gaussian_mean.npy"))[None].cuda()
        # class_idx = 95
        # G.target_classes.data = torch.tensor(class_idx).cuda()
        # with torch.no_grad():
        #     num_samples = 16384 #8192
        #     num_batches = 128
        #     z_orig = make_noise(num_samples, G.dim_z).cuda()
        #     orig_dists = torch.zeros(num_samples)
        #     batch_size = num_samples // num_batches
        #     for i in range(num_batches):
        #         orig_imgs = G(z_orig[i * batch_size: (i+1) * batch_size])
        #
        #         feats = inception(orig_imgs)[0].view(-1, 2048)
        #         orig_dists[i * batch_size: (i+1) * batch_size] = \
        #             ((target_feats - feats) ** 2).mean(-1).cpu()
        #     nearest_sample = orig_dists.argmin().item()
        #     print(nearest_sample, min(orig_dists), orig_dists[nearest_sample])
        #     z = z_orig[nearest_sample][None]
        #     print("Class idx ", class_idx,
        #           f" | Min: {min(orig_dists).item():.3}",
        #           f" | Mean: {orig_dists.mean().item():.3}")

        num_directions = 4
        target_feats = torch.tensor(np.load("stats/imagenet_gaussian_directions.npy"))[:num_directions].reshape(-1,
                                                                                                                2048).cuda()
        z = torch.zeros(num_directions * 8, 120).cuda()
        for sample_id in range(num_directions * 8):
            with torch.no_grad():
                num_samples = 8192
                num_batches = 64
                z_orig = make_noise(num_samples, G.dim_z).cuda()
                orig_dists = torch.zeros(num_samples)
                batch_size = num_samples // num_batches
                for i in range(num_batches):
                    orig_imgs = G(z_orig[i * batch_size: (i + 1) * batch_size])
                    orig_imgs = F.interpolate(orig_imgs, size=(299, 299),
                                              mode='bilinear', align_corners=False)
                    feats = inception(orig_imgs)
                    orig_dists[i * batch_size: (i + 1) * batch_size] = \
                        ((target_feats[sample_id][None] - feats) ** 2).mean(-1).cpu()
                nearest_sample = orig_dists.argmin().item()
                print(sample_id, nearest_sample, orig_dists[nearest_sample].item())
            z[sample_id] = z_orig[nearest_sample]
            torch.cuda.empty_cache()

        z_inv = nn.Parameter(z, requires_grad=True)
        optimizer = torch.optim.Adam([z_inv], lr=0.01)
        os.makedirs("inv_samples", exist_ok=True)

        for step in range(0, self.p.n_steps, 1):
            # if step == 10000:
            #     optimizer = torch.optim.LBFGS([z_inv], lr=0.8, max_iter=100, max_eval=None, tolerance_grad=1e-07,
            #                                   tolerance_change=1e-09, history_size=200)
            #
            # if isinstance(optimizer, torch.optim.LBFGS):
            #     def closure():
            #         optimizer.zero_grad()
            #         imgs_inv = G(z_inv)
            #         imgs_inv = F.interpolate(imgs_inv, size=(299, 299),
            #                                  mode='bilinear', align_corners=False)
            #         inv_feats = inception(imgs_inv)
            #         loss = ((target_feats - inv_feats) ** 2).mean()
            #         loss.backward()
            #         return loss
            #     optimizer.step(closure)
            optimizer.zero_grad()
            imgs_inv = G(z_inv)
            img_adv_feats = inception(imgs_inv)[0].view(-1, 2048)
            loss = ((target_feats - img_adv_feats) ** 2).mean()
            loss.backward()
            optimizer.step()

            if step % self.p.steps_per_log == 0:
                self.log(step, loss)
            if step % self.p.steps_per_save == 0:
                with torch.no_grad():
                    losses = ((target_feats - img_adv_feats) ** 2).mean(-1)
                # fig = plt.Figure(figsize=(8, 6))
                # ax = fig.add_subplot(1, 1, 1)
                # ax.imshow(to_image(imgs_inv))
                # ax.set_title(f"Mean | L2: {loss.item():.3}")
                # ax.axis("off")
                # fig_to_image(fig).save(f"inv_samples/gaussian_mean_{class_idx}_step{step}.png")

                fig, axes = plt.subplots(num_directions, 8, figsize=(24, 14))

                s = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
                for i in range(len(imgs_inv)):
                    axes[i // 8, i % 8].imshow(to_image(imgs_inv[i]))
                    axes[i // 8, i % 8].set_title(f"{s[i % 8]} $\lambda_{i // 8}$ | L2: {losses[i].item():.3}")
                    axes[i // 8, i % 8].axis('off')
                fig_to_image(fig).save(f"inv_samples/gaussian_directions_0_1_2_3_step{step}.png")
                plt.close(fig)

    # def train(self, G, inception):
    #     # transform = Compose([
    #     #     Resize(299),
    #     #     ToTensor(),
    #     #     Normalize(mean=[0.485, 0.456, 0.406],
    #     #               std=[0.229, 0.224, 0.225])
    #     # ])
    #
    #     # target_feats = torch.tensor(np.load("stats/imagenet_gaussian_mean.npy"))[None].cuda()
    #     self.p.batch_size = 30
    #     target_feats = torch.tensor(np.load("stats/imagenet_gaussian_directions.npy"))[:3].reshape(-1, 2048).cuda()
    #     G.cuda().eval()
    #
    #     z_orig = make_noise(self.p.batch_size, G.dim_z).cuda()
    #
    #     for i in range(self.p.batch_size):
    #         z_orig[i] = z_orig[0]
    #
    #     z_inv = nn.Parameter(z_orig, requires_grad=True)
    #     optimizer = torch.optim.Adam([z_inv], lr=0.001, betas=(0.9, 0.999))
    #
    #     os.makedirs("inv_samples", exist_ok=True)
    #     torch.save(z_orig, "inv_samples/orig_z.pt")
    #
    #     for step in range(0, self.p.n_steps, 1):
    #         G.zero_grad()
    #         optimizer.zero_grad()
    #
    #         imgs_inv = G(z_inv)
    #         imgs_adv = ((imgs_inv + 1.) / 2.).clamp(0, 1)
    #         imgs_adv = F.interpolate(imgs_adv, size=(299, 299),
    #                                  mode='bilinear', align_corners=False)
    #         mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
    #         std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    #         imgs_adv = (imgs_adv - mean[..., None, None]) / std[..., None, None]
    #
    #         img_adv_feats = inception(imgs_adv)
    #         loss = ((target_feats - img_adv_feats) ** 2).mean()
    #         loss.backward()
    #         optimizer.step()
    #
    #         if step % self.p.steps_per_log == 0:
    #             self.log(step, loss)
    #         if step % self.p.steps_per_save == 0:
    #             torch.save(z_inv.cpu().data, f"inv_samples/direction_0_1_2_inv_z_{step}.pt")
    #             # fig = plt.Figure(figsize=(8, 6))
    #             # ax = fig.add_subplot(1, 1, 1)
    #             # ax.imshow(to_image(imgs_inv))
    #             # ax.set_title(f"Mean Inversion")
    #             fig, axes = plt.subplots(3, self.p.batch_size // 3, figsize=(24, 8))
    #             for i in range(len(imgs_adv)):
    #                 axes[i // 10, i % 10].imshow(to_image(imgs_inv[i]))
    #                 # axes[i // 10, i % 10].set_title(f"Inversion {i}")
    #
    #             fig_to_image(fig).save(f"inv_samples/direction_0_1_2_step{step}.png")
    #             plt.close(fig)
    #
    #         if (step + 1) % (self.p.n_steps - 1000) == 0:
    #             print((step + 1),(self.p.n_steps - 1000))
    #             for param_group in optimizer.param_groups:
    #                 param_group['lr'] = 1e-4



