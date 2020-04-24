import os
import json
import torch
from collections import OrderedDict

from models.gan_load import make_big_gan, make_proggan, make_external
from constants import WEIGHTS


def load_from_dir(root_dir, model_index=None, G_weights=None, verbose=False):
    args = json.load(open(os.path.join(root_dir, 'args.json')))

    models_dir = os.path.join(root_dir, 'models')
    if model_index is None:
        models = os.listdir(models_dir)
        model_index = max(
            [int(name.split('.')[0].split('_')[-1]) for name in models
             if name.startswith('deformator')])

        if verbose:
            print('using max index {}'.format(model_index))


    if G_weights is None:
        G_weights = args['gan_weights']
    if G_weights is None or not os.path.isfile(G_weights):
        if verbose:
            print('Using default local G weights')
        G_weights = WEIGHTS[args['gan_type']]

    if args['gan_type'] == 'BigGAN':
        G = make_big_gan(G_weights, args['target_class']).eval()
    elif args['gan_type'] in ['ProgGAN', 'PGGAN']:
        G = make_proggan(G_weights)
    else:
        G = make_external(G_weights)

    return G.eval().cuda()
