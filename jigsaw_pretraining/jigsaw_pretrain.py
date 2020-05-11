import os
import random

import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from data_helper import JigsawDataset
from data_loader_single_map import LabeledDataset
from helper import collate_fn, draw_box

import torchvision.models as tvmodel
import torch.nn as nn
import torch

from itertools import permutations
import random


class res_encoder(nn.Module):
    def __init__(self):
        super(res_encoder, self).__init__()
        resnet = tvmodel.resnet18(pretrained=False)
        resnet_out_channel = resnet.fc.in_features
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        self.Conn_layers1 = nn.Sequential(
            nn.Linear(4 * 4 * 1024, 4096),
            nn.LeakyReLU())

        self.Conn_layers2 = nn.Sequential(
            nn.Linear(4096 * 6, 50 * 50),
            nn.LeakyReLU())

    def forward(self, input):
        x1 = self.resnet(input[:, 0])
        x1 = self.Conv_layers(x1)
        x1 = x1.view(x1.size()[0], -1)
        x1 = self.Conn_layers1(x1)

        x2 = self.resnet(input[:, 1])
        x2 = self.Conv_layers(x2)
        x2 = x2.view(x2.size()[0], -1)
        x2 = self.Conn_layers1(x2)
        x3 = self.resnet(input[:, 2])
        x3 = self.Conv_layers(x3)
        x3 = x3.view(x3.size()[0], -1)
        x3 = self.Conn_layers1(x3)

        x4 = self.resnet(input[:, 3])
        x4 = self.Conv_layers(x4)
        x4 = x4.view(x4.size()[0], -1)
        x4 = self.Conn_layers1(x4)

        x5 = self.resnet(input[:, 4])
        x5 = self.Conv_layers(x5)
        x5 = x5.view(x5.size()[0], -1)
        x5 = self.Conn_layers1(x5)

        x6 = self.resnet(input[:, 5])
        x6 = self.Conv_layers(x6)
        x6 = x6.view(x6.size()[0], -1)
        x6 = self.Conn_layers1(x6)

        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)

        x = self.Conn_layers2(x)
        return x



class jigsaw_classifier(nn.Module):
    def __init__(self):
        super(jigsaw_classifier, self).__init__()
        self.encoder = res_encoder()
        self.hidden_layers = nn.Sequential(
            nn.Linear(50*50, 1024),
            nn.Linear(1024, 20))

    def forward(self, x):
        x = self.encoder(x)
        x = self.hidden_layers(x)
        return x
        #return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda'
    epoch = 2
    batchsize = 2
    lr = 0.00001
    transform = torchvision.transforms.ToTensor()

    image_folder = '../data'
    annotation_csv = '../data/annotation.csv'

    unlabeled_scene_index = np.arange(86)
    unlabeled_scene_index_val = np.arange(86, 106)

    l = list(permutations(range(6)))
    permutation_list = random.sample(l,20)

    #pre-training
    unlabeled_trainset = JigsawDataset(image_folder=image_folder,
                                          first_dim='sample',
                                          scene_index=unlabeled_scene_index,
                                          transform=transform,
                                          permutation_list=permutation_list
                                          )
    un_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=batchsize, shuffle=True, num_workers=1)

    unlabeled_valset = JigsawDataset(image_folder=image_folder,
                                        first_dim='sample',
                                        scene_index=unlabeled_scene_index_val,
                                        transform=transform,
                                        permutation_list=permutation_list
                                        )
    un_valloader = torch.utils.data.DataLoader(unlabeled_valset, batch_size=batchsize, shuffle=True, num_workers=2)

    model_ae = jigsaw_classifier().to(device)
    criterion_ae = nn.CrossEntropyLoss()
    optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=lr)
    for ep in range(epoch):
        model_ae.train()
        num_correct=0
        yl = torch.Tensor([0]).cuda()
        for it, (image_tensor, image_permutated, label) in enumerate(un_trainloader):
            sample = image_permutated.numpy().transpose(3, 4, 0, 1, 2).reshape(256, 306, -1)
            sample = cv2.resize(sample, (256, 256))
            sample = sample.reshape(256, 256, batchsize, 6, 3).transpose(2, 3, 4, 0, 1)
            sample = torch.Tensor(sample).to(device)
            y_pred = model_ae(sample)
            label = label.to(device)
            loss = criterion_ae(y_pred, label)
            score, predicted = torch.max(y_pred, 1)
            num_correct += (label == predicted).sum().item()
            optimizer_ae.zero_grad()
            loss.backward()
            optimizer_ae.step()
            yl = yl + loss
            if it%500 == 0:
                print("Epoch %d/%d| Step %d/%d| Loss: %.4f | Acc: %.2f " % (ep, epoch, it, len(unlabeled_trainset) // batchsize, loss, float(num_correct)/(len(un_trainloader)*batchsize)))
        torch.save(model_ae.encoder.state_dict(), "models_pkl/20_resencoder_bbox_segment_epoch" + str(ep + 1) + ".pkl")
        torch.save(optimizer_ae.state_dict(), "models_pkl/20_resencoder_bbox_segment_optimizer_epoch" + str(ep + 1) + ".pkl")

        model_ae.eval()
        yt = torch.Tensor([0]).cuda()
        num_correct_val = 0
        with torch.no_grad():
            for it, (image_tensor, image_permutated, label) in enumerate(un_valloader):
                sample =  image_permutated.numpy().transpose(3, 4, 0, 1, 2).reshape(256, 306, -1)
                sample = cv2.resize(sample, (256, 256))
                sample = sample.reshape(256, 256, batchsize, 6, 3).transpose(2, 3, 4, 0, 1)
                sample = torch.Tensor(sample).to(device)
                output = model_ae(sample)
                score, predicted = torch.max(output, 1)
                label = label.to(device)
                loss = criterion_ae(output, label)
                num_correct_val += (label == predicted).sum().item()
                yt = yt + loss
            print("Epoch %d/%d|val loss: %.2f |acc: %.2f" % (ep, epoch, yt/(len(un_valloader)*batchsize), float(num_correct_val)/(len(un_valloader)*batchsize)))




