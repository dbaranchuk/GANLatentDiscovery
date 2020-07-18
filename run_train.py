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
from efros_networks.resnet import resnet50 as efros_resnet50

from lib.gan_model.model import Generator
from latent_deformator import LatentDeformator
from latent_shift_predictor import ResNetPredictor, SiameseResNetPredictor
from constants import DEFORMATOR_TYPE_DICT, DEFORMATOR_LOSS_DICT, SHIFT_DISTRIDUTION_DICT, WEIGHTS


##################################################
# import signal
#
# #Close session
# def handler(signum, frame):
#     raise Exception('Action took too much time')
#
# signal.signal(signal.SIGALRM, handler)
# signal.alarm(3)
#
# try:
#     from lib.gan_model.model import Generator
# except:
#     from lib.gan_model.model import Generator
# signal.alarm(0)
################################################

def main():
    parser = argparse.ArgumentParser(description='Latent space rectification')
    for key, val in Params().__dict__.items():
        parser.add_argument('--{}'.format(key), type=type(val), default=val)

    parser.add_argument('--args', type=str, default=None, help='json with all arguments')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--gan_type', type=str, choices=WEIGHTS.keys())
    parser.add_argument('--gan_weights', type=str, default=None)
    parser.add_argument('--target_class', type=int, default=239)
    parser.add_argument('--json', type=str)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--deformator', type=str, default='ortho',
                        choices=DEFORMATOR_TYPE_DICT.keys())
    parser.add_argument('--deformator_random_init', type=bool, default=False)

    parser.add_argument('--predictor_size', type=int, default=224)
    parser.add_argument('--predictor', type=str,
                        choices=['ResNet', 'SiameseResNet'], default='ResNet')
    parser.add_argument('--distribution_key', type=str,
                        choices=SHIFT_DISTRIDUTION_DICT.keys())

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
        G = make_big_gan(weights_path, args.target_class).eval()
    elif args.gan_type == 'ProgGAN':
        G = make_proggan(weights_path).eval()
    elif args.gan_type == 'StyleGAN2':
        pretrained_model = torch.load(weights_path)
        G = Generator(256, 512, 8, channel_multiplier=2)
        G.load_state_dict(pretrained_model['g_ema'], strict=False)
        G.train(False)
        G.dim_z = G.style_dim
        for param in G.parameters():
            param.requires_grad = False
    else:
        G = make_external(weights_path).eval()

    deformator = LatentDeformator(G.dim_z,
                                  type=DEFORMATOR_TYPE_DICT[args.deformator],
                                  random_init=args.deformator_random_init).cuda()

    if args.predictor == 'ResNet':
        predictor = ResNetPredictor(args.max_latent_ind, args.predictor_size).cuda()
    elif args.predictor == 'SiameseResNet':
        predictor = SiameseResNetPredictor(args.max_latent_ind, args.predictor_size).cuda()
    else:
        raise Exception("Unknown predictor type")
    # inception = inception_v3(num_classes=1000, aux_logits=False, pretrained=True).cuda().eval()
    # inception.fc = torch.nn.Identity()

    efros_model = efros_resnet50(num_classes=1)
    state_dict = torch.load('efros_weights/blur_jpg_prob0.5.pth', map_location='cpu')
    efros_model.load_state_dict(state_dict['model'])
    efros_model.cuda().eval()
    for param in efros_model.parameters():
        param.requires_grad = False

    # training
    print("Start training...")
    trainer = Trainer(params=Params(**args.__dict__), out_dir=args.out)

    if args.mode == 'train':
        trainer.train(G, deformator, predictor, efros_model)


if __name__ == '__main__':
    main()
