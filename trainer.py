import os
import torch
from utils import make_noise
from torch_tools.visualization import to_image
from matplotlib import pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
from visualization import fig_to_image


from matplotlib.backends.backend_pdf import PdfPages
plt.style.use('seaborn-whitegrid')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


class Params(object):
    def __init__(self, **kwargs):
        self.n_steps = 301
        self.batch_size = 32


        self.steps_per_log = 300
        self.steps_per_save = 300
        self.l2_loss_weight = 10

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

    def log(self, step, img_l2_loss, loss):
            print(f'Step {step} pixel_loss: {img_l2_loss.item():.3} fake_score: {loss.item():.3}')

    def train(self, G, model, class_idx):
        # trans = transforms.Compose([
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        mean = torch.tensor([0.485, 0.456, 0.406]).cuda()[..., None, None]
        std = torch.tensor([0.229, 0.224, 0.225]).cuda()[..., None, None]

        sample_dir = f'efros_dataset_200/val/{class_idx}'
        os.makedirs("efros_dataset_200", exist_ok=True)
        os.makedirs("efros_dataset_200/val", exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

        orig_sample_dir = f'orig_efros_dataset_200/val/{class_idx}'
        os.makedirs("orig_efros_dataset_200", exist_ok=True)
        os.makedirs("orig_efros_dataset_200/val", exist_ok=True)
        os.makedirs(orig_sample_dir, exist_ok=True)

        G.target_classes.data = torch.tensor(class_idx).cuda()

        for batch_id in range(2):
            print(class_idx, batch_id)
            z_orig = make_noise(self.p.batch_size, G.dim_z).cuda()
            z_adv = nn.Parameter(z_orig, requires_grad=True)
            optimizer = torch.optim.Adam([z_adv], lr=0.003, betas=(0.9, 0.999))

            # Original samples
            orig_samples = G(z_orig)

            for step in range(0, self.p.n_steps, 1):
                G.zero_grad()
                optimizer.zero_grad()

                imgs_efros = G(z_adv)
                imgs_adv = ((imgs_efros + 1.) / 2.).clamp(0, 1)
                imgs_adv = F.interpolate(imgs_adv, size=(224, 224),
                                         mode='bilinear', align_corners=False)
                imgs_adv = (imgs_adv - mean) / std

                ####################
                probs = model(imgs_adv).sigmoid()
                pixel_loss = self.p.l2_loss_weight * ((orig_samples - imgs_efros) ** 2).mean()
                loss = probs.mean() + pixel_loss
                loss.backward()
                optimizer.step()

                if step == 0:
                    zero_step_probs = probs.detach()
                if step % self.p.steps_per_log == 0:
                    self.log(step, pixel_loss, loss)
                if step > 0 and step % self.p.steps_per_save == 0:
                    for i in range(self.p.batch_size):
                        img = to_image(imgs_efros[i])
                        img.save(os.path.join(sample_dir, f'{batch_id * 25 + i}.png'))

                        orig_img = to_image(orig_samples[i])
                        orig_img.save(os.path.join(orig_sample_dir, f'{batch_id * 25 + i}.png'))

                    with PdfPages(f"efros_samples/{class_idx}_{batch_id}_step{step}.pdf") as pdf:
                        fig, axes = plt.subplots(len(imgs_efros), 3, figsize=(20, 200))
                        for i in range(len(imgs_efros)):
                            axes[i][0].imshow(to_image(orig_samples[i]))
                            axes[i][0].set_title(f"Original Sample Prob: {zero_step_probs[i].item():.2}", fontsize=12)
                            axes[i][0].axis('off')
                            axes[i][0].grid(False)

                            axes[i][1].imshow(to_image(imgs_efros[i]))
                            axes[i][1].set_title(f"After Prob: {probs[i].item():.2}", fontsize=12)
                            axes[i][1].axis('off')
                            axes[i][1].grid(False)

                            diff_image = (imgs_efros[i] - orig_samples[i]).mean(0).cpu().detach()
                            axes[i][2].imshow(diff_image)
                            axes[i][2].set_title("Difference", fontsize=12)
                            axes[i][2].axis('off')
                            axes[i][2].grid(False)

                        pdf.savefig(fig, bbox_inches='tight')
                        # pdf.close()
                        # fig_to_image(fig).save(f"efros_samples/step{step}.png")
                        # plt.close(fig)




