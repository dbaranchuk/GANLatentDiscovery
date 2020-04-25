import os
import torch
from torch import nn
from utils import make_noise
from torch_tools.visualization import to_image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

    def log(self, step, img_l2_loss, img_feat_l2_loss):
            print('Step {} img_l2_loss: {:.3} img_feat_l2_loss: {:.3}'.format(step, img_l2_loss.item(),
                                                                              img_feat_l2_loss.item()))

    def train(self, G, inception):
        G.cuda().eval()

        z_orig = make_noise(self.p.batch_size, G.dim_z).cuda()
        z_adv = nn.Parameter(z_orig, requires_grad=True)
        optimizer = torch.optim.Adam([z_adv], lr=0.001, betas=(0.9, 0.999))

        imgs = G(z_orig).detach()
        img_feats = inception(((imgs + 1.) / 2.).clamp(0, 1))
        if isinstance(img_feats, list):
            img_feats = img_feats[0].detach()

        os.makedirs("adv_samples", exist_ok=True)

        for step in range(0, self.p.n_steps, 1):
            G.zero_grad()
            optimizer.zero_grad()

            imgs_adv = G(z_adv + 1e-6)
            imgs_loss = self.p.l2_loss_weight * ((imgs - imgs_adv) ** 2).mean()

            img_adv_feats = inception(((imgs_adv + 1.) / 2.).clamp(0, 1))
            if isinstance(img_adv_feats, list):
                img_adv_feats = img_adv_feats[0]

            l2 = ((img_feats - img_adv_feats) ** 2).mean()
            inception_loss = l2

            loss = imgs_loss - inception_loss
            loss.backward()
            optimizer.step()

            if step % self.p.steps_per_log == 0:
                self.log(step, imgs_loss, inception_loss)
            if step % self.p.steps_per_save == 0:
                for i in range(len(imgs_adv)):

                    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

                    axes[0].imshow(to_image(imgs[i]))
                    axes[0].set_title("Original")

                    axes[1].imshow(to_image(imgs_adv[i]))
                    axes[1].set_title("Adversarial")

                    diff_image = (imgs_adv[i] - imgs[i]).mean(0).cpu().detach()
                    diff_img = axes[2].imshow(diff_image, cmap='viridis')
                    fig.colorbar(diff_img, cax=axes[2], orientation='vertical')
                    axes[2].set_title("Difference")

                    fig_to_image(fif).save(f"adv_samples/{i}_step{step}.png")




