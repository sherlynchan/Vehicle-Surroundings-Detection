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

from data_loader_single_map import LabeledDataset
from helper import collate_fn, draw_box

import torchvision.models as tvmodel
import torch.nn as nn
import torch


class res_roadmap(nn.Module):
    def __init__(self):
        super(res_roadmap, self).__init__()
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
        self.deconv_layers = nn.Sequential(
            nn.Conv2d(1, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        # 以下是YOLOv1的最后2个全连接层
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
        x = x.view(-1, 1, 50, 50)
        x = self.deconv_layers(x)

        return x.view(-1, 800*800)


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = 'cuda'

    image_folder = 'data'
    annotation_csv = 'data/annotation.csv'

    unlabeled_scene_index = np.arange(106)
    labeled_scene_index = np.arange(106, 126)
    labeled_scene_index_val = np.arange(126, 134)

    epoch = 50
    batchsize = 1
    lr = 0.00001
    transform = torchvision.transforms.ToTensor()
    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index,
                                      transform=transform
                                      )
    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=batchsize, shuffle=True, num_workers=1,
                                              collate_fn=collate_fn)

    labeled_valset = LabeledDataset(image_folder=image_folder,
                                    annotation_file=annotation_csv,
                                    scene_index=labeled_scene_index_val,
                                    transform=transform
                                    )
    valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=batchsize, shuffle=True, num_workers=2,
                                            collate_fn=collate_fn)
    print('1')
    model = res_roadmap().to(device)
    model.load_state_dict(torch.load('/scratch/sg5722/models_pkl/bbox_segment_epoch1.pkl'))


    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(torch.load('/scratch/sg5722/models_pkl/bbox_segment_optimizer_epoch1.pkl'))
    print('2')
    #torch.save(model.state_dict(), "/scratch/sg5722/models_pkl/bbox_segment_epoch" + str(0) + ".pkl")
    #torch.save(optimizer.state_dict(), "/scratch/sg5722/models_pkl/bbox_segment_optimizer_epoch" + str(0) + ".pkl")
    for ep in range(1, 30):
        model.train()
        yl = torch.Tensor([0]).cuda()
        for it, (sample, box, box_tensor, road_image, road_image_tensor) in enumerate(trainloader):
            sample = torch.stack(sample)
            sample = sample.numpy().transpose(3, 4, 0, 1, 2).reshape(256, 306, -1)
            sample = cv2.resize(sample, (256, 256))
            sample = sample.reshape(256, 256, batchsize, 6, 3).transpose(2, 3, 4, 0, 1)
            sample = torch.Tensor(sample).to(device)
            pred = model(sample)
            labels = torch.Tensor(box[0].reshape(-1,800*800)+0).float().to(device)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if it%500 == 0:
                print("Epoch %d/%d| Step %d/%d| Loss: %.2f" % (ep, epoch, it, len(labeled_trainset) // batchsize, loss))
            yl = yl + loss
        torch.save(model.state_dict(), "/scratch/sg5722/models_pkl/bbox_segment_epoch" + str(ep + 1) + ".pkl")
        torch.save(optimizer.state_dict(), "/scratch/sg5722/models_pkl/bbox_segment_optimizer_epoch" + str(ep + 1) + ".pkl")

        model.eval()
        yt = torch.Tensor([0]).cuda()
        with torch.no_grad():
            for it, (sample, box, box_tensor, road_image, road_image_tensor) in enumerate(valloader):
                sample = torch.stack(sample)
                sample = sample.numpy().transpose(3, 4, 0, 1, 2).reshape(256, 306, -1)
                sample = cv2.resize(sample, (256, 256))
                sample = sample.reshape(256, 256, batchsize, 6, 3).transpose(2, 3, 4, 0, 1)
                sample = torch.Tensor(sample).to(device)
                pred = model(sample)
                labels = torch.Tensor(box[0].reshape(-1,800*800)+0).float().to(device)
                loss = criterion(pred, labels)*batchsize
                yt = yt + loss
            print("Epoch %d/%d|val loss: %.2f" % (ep, epoch, yt/len(valloader)))