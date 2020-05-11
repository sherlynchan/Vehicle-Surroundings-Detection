import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box, compute_ts_road_map
import pdb

import torchvision.models as tvmodel
import torch.nn as nn
import torch


class ResNetVAE(nn.Module):
    def __init__(self):
        super(ResNetVAE, self).__init__()

        resnet = tvmodel.resnet18(pretrained=True)
        self.base_layers = nn.Sequential(*list(resnet.children())[:-2], nn.AdaptiveAvgPool2d(output_size=(8, 8)),)
        # pt_path = 'models_pkl/SSL_0507_model.pkl'
        # if os.path.isfile(pt_path):
        #     pt_model = ResNetSSL()
        #     pt_model.load_state_dict(torch.load(pt_path))
        #     self.base_layers.load_state_dict(pt_model.base_layers.state_dict())
        #     print('load weights from pretrained model')

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3072, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(64, 1024, kernel_size=4, stride=3), ## 8, 1024, 25, 25
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),  ## 50
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  ## 100
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  ## 200
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  ##400
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),  ##800
                nn.Sigmoid()
            )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input):
        num_views = input.size()[1]
        encoders = []
        for i in range(num_views):
            cur = input[:,i] ## BS * 3 * 256 * 306
            cur = self.base_layers(cur) ## BS * 512 * 8 * 10 -> BS * 512 * 8 * 8
            encoders.append(cur)
        hidden = torch.cat(encoders, dim=1) ## BS * 3072 * 8 * 8
        hidden = self.conv_layers(hidden) ## BS * 1024 * 8 * 8
        mu_logvar = hidden.view(-1, 2, 4096)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        output = z.view(-1, 64, 8, 8)
        output = self.deconv_layers(output)
        return output.squeeze(), mu, logvar
