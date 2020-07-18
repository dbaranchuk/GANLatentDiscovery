import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34, resnet50
import numpy as np


def save_hook(module, input, output):
    setattr(module, 'output', output)


class SiameseResNetPredictor(nn.Module):
    def __init__(self, dim, downsample=None):
        super(ResNetPredictor, self).__init__()
        self.features_extractor = resnet50(pretrained=False)
        # self.features_extractor.conv1 = nn.Conv2d(
        #     6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
        #                         mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(self.features_extractor.fc.weight.shape[1], np.product(dim))
        self.shift_estimator = nn.Linear(self.features_extractor.fc.weight.shape[1], 1)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x1, x2 = F.interpolate(x1, self.downsample), F.interpolate(x2, self.downsample)

        self.features_extractor(x1)
        features1 = self.features.output.view([batch_size, -1])
        self.features_extractor(x2)
        features2 = self.features.output.view([batch_size, -1])

        features = features1 + features2
        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)

        return logits, shift.squeeze()


class ResNetPredictor(nn.Module):
    def __init__(self, dim, downsample=None):
        super(ResNetPredictor, self).__init__()
        self.features_extractor = resnet50(pretrained=False)
        self.features_extractor.conv1 = nn.Conv2d(
            6, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(self.features_extractor.fc.weight.shape[1], np.product(dim))
        self.shift_estimator = nn.Linear(self.features_extractor.fc.weight.shape[1], 1)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x1, x2 = F.interpolate(x1, self.downsample), F.interpolate(x2, self.downsample)
        self.features_extractor(torch.cat([x1, x2], dim=1))
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)

        return logits, shift.squeeze()