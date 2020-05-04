import os
import sys
import argparse
import json
import random
import torch

import matplotlib
matplotlib.use("Agg")

import numpy as np
from models.gan_load import make_big_gan, make_proggan, make_external
from trainer import Trainer, Params
from inception import InceptionV3
from torchvision.models import inception_v3

WEIGHTS = {
    'BigGAN': 'models/pretrained/BigGAN/138k/G_ema.pth',
    'ProgGAN': 'models/pretrained/ProgGAN/100_celeb_hq_network-snapshot-010403.pth',
    'SN_MNIST': 'models/pretrained/GANs/SN_MNIST',
    'Anime_64': 'models/pretrained/GANs/SN_Anime',
}


def main():
    parser = argparse.ArgumentParser(description='Latent space rectification')
    for key, val in Params().__dict__.items():
        parser.add_argument('--{}'.format(key), type=type(val), default=None)

    parser.add_argument('--args', type=str, default=None, help='json with all arguments')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--gan_type', type=str, choices=WEIGHTS.keys())
    parser.add_argument('--gan_weights', type=str, default=None)
    parser.add_argument('--target_class', type=int, default=239)
    parser.add_argument('--json', type=str)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if args.args is not None:
        with open(args.args) as args_json:
            args_dict = json.load(args_json)
            args.__dict__.update(**args_dict)

    # save run params
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    with open(os.path.join(args.out, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')

    # init models
    if args.gan_weights is not None:
        weights_path = args.gan_weights
    else:
        weights_path = WEIGHTS[args.gan_type]

    if args.gan_type == 'BigGAN':
        G = make_big_gan(weights_path, args.target_class).cuda().eval()
    elif args.gan_type == 'ProgGAN':
        G = make_proggan(weights_path).cuda().eval()
    else:
        G = make_external(weights_path).cuda().eval()

    inception = inception_v3(num_classes=1000, aux_logits=False,
                             pretrained=True, transform_input=False).cuda().eval()
    inception.fc = torch.nn.Identity()
    for param in inception.parameters():
        param.requires_grad = False

    for param in G.parameters():
        param.requires_grad = False

    # training
    trainer = Trainer(params=Params(**args.__dict__), out_dir=args.out)

    if args.mode == 'train':
        trainer.train(G, inception)


if __name__ == '__main__':
    main()
