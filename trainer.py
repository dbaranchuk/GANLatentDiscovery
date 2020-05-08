import os
import torch
from utils import make_noise
from torch_tools.visualization import to_image
from matplotlib import pyplot as plt

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

        self.l2_loss_weight = 10.0

        self.steps_per_log = 100
        self.steps_per_save = 2500

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
            print('Step {} Loss: {:.3} '.format(step, loss.item()))

    def train(self, G, model):
        # trans = transforms.Compose([
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        G.cuda().eval()

        z_orig = make_noise(self.p.batch_size, G.dim_z).cuda()
        z_adv = nn.Parameter(z_orig + 1e-6, requires_grad=True)
        optimizer = torch.optim.Adam([z_adv], lr=0.003, betas=(0.9, 0.999))

        # Original samples
        orig_samples = G(z_orig)
        os.makedirs("efros_samples", exist_ok=True)
        torch.save(z_orig, "efros_samples/orig_z.pt")

        for step in range(0, self.p.n_steps, 1):
            G.zero_grad()
            optimizer.zero_grad()

            imgs_efros = G(z_adv)
            imgs_adv = ((imgs_efros + 1.) / 2.).clamp(0, 1)
            imgs_adv = F.interpolate(imgs_adv, size=(224, 224),
                                     mode='bilinear', align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
            std = torch.tensor([0.229, 0.224, 0.225]).cuda()
            imgs_adv = (imgs_adv - mean[..., None, None]) / std[..., None, None]

            ####################
            loss = model(imgs_adv).sigmoid().log().mean()
            loss.backward()
            optimizer.step()

            if step % self.p.steps_per_log == 0:
                self.log(step, loss)
            if step % self.p.steps_per_save == 0:
                torch.save(z_adv.data, f"efros_samples/efros_z_{step}.pt")
                for i in range(len(imgs_efros)):

                    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

                    axes[0].imshow(to_image(orig_samples[i]))
                    axes[0].set_title("Original")

                    axes[1].imshow(to_image(imgs_efros[i]))
                    axes[1].set_title("Adversarial")

                    diff_image = (imgs_efros[i] - orig_samples[i]).mean(0).cpu().detach()
                    axes[2].imshow(diff_image)
                    axes[2].set_title("Difference")

                    fig_to_image(fig).save(f"efros_samples/{i}_step{step}.png")
                    plt.close(fig)




