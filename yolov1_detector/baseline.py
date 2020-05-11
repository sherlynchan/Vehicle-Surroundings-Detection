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

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box

import torchvision.models as tvmodel
import torch.nn as nn
import torch



def convert(size, box):
    """将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
    并进行归一化"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[1,1]*10 + box[1,0]*10) / 2.0
    y = (box[0,1]*10 + box[0,2]*10) / 2.0
    w = abs(box[1,1]*10 - box[1,0]*10)
    h = abs(box[0,2]*10 - box[0,1]*10)
    x = x * dw +0.5
    w = w * dw
    y = y * dh + 0.5
    h = h * dh
    return (x, y, w, h)

NUM_BBOX = 2
CLASSES = ['other_vehicle', 'bicycle','car', 'pedestrian', 'truck', 'bus', 'motorcycle', 'emergency_vehicle', 'animal']
def convert_bbox2labels(bbox):
    """将bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
    注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式"""
    gridsize = 1.0/7
    labels = np.zeros((7,7,5*NUM_BBOX+len(CLASSES)))  # 注意，此处需要根据不同数据集的类别个数进行修改
    for i in range(len(bbox)//5):
        gridx = int(bbox[i*5+1] // gridsize)  # 当前bbox中心落在第gridx个网格,列
        gridy = int(bbox[i*5+2] // gridsize)  # 当前bbox中心落在第gridy个网格,行
        # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
        gridpx = bbox[i * 5 + 1] / gridsize - gridx
        gridpy = bbox[i * 5 + 2] / gridsize - gridy
        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 10+int(bbox[i*5])] = 1
    return labels

class Loss_yolov1(nn.Module):
    def __init__(self):
        super(Loss_yolov1,self).__init__()

    def forward(self, pred, labels):
        """
        :param pred: (batchsize,30,7,7)的网络输出数据
        :param labels: (batchsize,30,7,7)的样本标签数据
        :return: 当前批次样本的平均损失
        """
        num_gridx, num_gridy = labels.size()[-2:]  # 划分网格数量
        num_b = 2  # 每个网格的bbox数量
        num_cls = 9  # 类别数量
        noobj_confi_loss = 0.  # 不含目标的网格损失(只有置信度损失)
        coor_loss = 0.  # 含有目标的bbox的坐标损失
        obj_confi_loss = 0.  # 含有目标的bbox的置信度损失
        class_loss = 0.  # 含有目标的网格的类别损失
        n = labels.size()[0]  # batchsize的大小

        # 可以考虑用矩阵运算进行优化，提高速度，为了准确起见，这里还是用循环
        for i in range(n):  # batchsize循环
            for m in range(7):  # x方向网格循环
                for n in range(7):  # y方向网格循环
                    if labels[i,4,m,n]==1:# 如果包含物体
                        # 将数据(px,py,w,h)转换为(x1,y1,x2,y2)
                        # 先将px,py转换为cx,cy，即相对网格的位置转换为标准化后实际的bbox中心位置cx,xy
                        # 然后再利用(cx-w/2,cy-h/2,cx+w/2,cy+h/2)转换为xyxy形式，用于计算iou
                        bbox1_pred_xyxy = ((pred[i,0,m,n]+m)/num_gridx - pred[i,2,m,n]/2,(pred[i,1,m,n]+n)/num_gridy - pred[i,3,m,n]/2,
                                           (pred[i,0,m,n]+m)/num_gridx + pred[i,2,m,n]/2,(pred[i,1,m,n]+n)/num_gridy + pred[i,3,m,n]/2)
                        bbox2_pred_xyxy = ((pred[i,5,m,n]+m)/num_gridx - pred[i,7,m,n]/2,(pred[i,6,m,n]+n)/num_gridy - pred[i,8,m,n]/2,
                                           (pred[i,5,m,n]+m)/num_gridx + pred[i,7,m,n]/2,(pred[i,6,m,n]+n)/num_gridy + pred[i,8,m,n]/2)
                        bbox_gt_xyxy = ((labels[i,0,m,n]+m)/num_gridx - labels[i,2,m,n]/2,(labels[i,1,m,n]+n)/num_gridy - labels[i,3,m,n]/2,
                                        (labels[i,0,m,n]+m)/num_gridx + labels[i,2,m,n]/2,(labels[i,1,m,n]+n)/num_gridy + labels[i,3,m,n]/2)
                        iou1 = calculate_iou(bbox1_pred_xyxy,bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy,bbox_gt_xyxy)
                        # 选择iou大的bbox作为负责物体
                        if iou1 >= iou2:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i,0:2,m,n] - labels[i,0:2,m,n])**2) \
                                        + torch.sum((pred[i,2:4,m,n].sqrt()-labels[i,2:4,m,n].sqrt())**2))
                            obj_confi_loss = obj_confi_loss + (pred[i,4,m,n] - iou1)**2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i,9,m,n]-iou2)**2)
                        else:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i,5:7,m,n] - labels[i,5:7,m,n])**2) \
                                        + torch.sum((pred[i,7:9,m,n].sqrt()-labels[i,7:9,m,n].sqrt())**2))
                            obj_confi_loss = obj_confi_loss + (pred[i,9,m,n] - iou2)**2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中,注意，对于标签的置信度应该是iou1
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i, 4, m, n]-iou1) ** 2)
                        class_loss = class_loss + torch.sum((pred[i,10:,m,n] - labels[i,10:,m,n])**2)
                    else:  # 如果不包含物体
                        noobj_confi_loss = noobj_confi_loss + 0.5 * torch.sum(pred[i,[4,9],m,n]**2)

        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss
        # 此处可以写代码验证一下loss的大致计算是否正确，这个要验证起来比较麻烦，比较简洁的办法是，将输入的pred置为全1矩阵，再进行误差检查，会直观很多。
        return loss/n

def calculate_iou(bbox1,bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的交集
    if bbox1[2]<bbox2[0] or bbox1[0]>bbox2[2] or bbox1[3]<bbox2[1] or bbox1[1]>bbox2[3]:
        pass
    else:
        intersect_bbox[0] = max(bbox1[0],bbox2[0])
        intersect_bbox[1] = max(bbox1[1],bbox2[1])
        intersect_bbox[2] = min(bbox1[2],bbox2[2])
        intersect_bbox[3] = min(bbox1[3],bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # input()

    if area_intersect>0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0


class YOLOv1_resnet(nn.Module):
    def __init__(self):
        super(YOLOv1_resnet, self).__init__()
        resnet = tvmodel.resnet18(pretrained=True)  # 调用torchvision里的resnet34预训练模型
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两层
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
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
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers1 = nn.Sequential(
            nn.Linear(4 * 4 * 1024, 4096),
            nn.LeakyReLU())
        self.Conn_layers2 = nn.Sequential(
            nn.Linear(4096 * 6, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 7 * 7 * (5 * NUM_BBOX + len(CLASSES))),
            nn.Sigmoid())

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

        return x.reshape(-1, (5 * NUM_BBOX + len(CLASSES)), 7, 7)

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
                                      transform=transform,
                                      extra_info=True
                                      )
    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=batchsize, shuffle=True, num_workers=1,
                                              collate_fn=collate_fn)

    labeled_valset = LabeledDataset(image_folder=image_folder,
                                    annotation_file=annotation_csv,
                                    scene_index=labeled_scene_index_val,
                                    transform=transform,
                                    extra_info=True
                                    )
    valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=batchsize, shuffle=True, num_workers=2,
                                            collate_fn=collate_fn)
    model = YOLOv1_resnet().to(device)
    criterion = Loss_yolov1()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torch.save(model.state_dict(), "models_pkl/YOLOv1_epoch_model" + str(0) + ".pkl")
    torch.save(optimizer.state_dict(), "models_pkl/YOLOv1_epoch_optimizer" + str(0) + ".pkl")
    for ep in range(epoch):
        model.train()
        yl = torch.Tensor([0]).cuda()
        for it, (sample, target, road_image, extra) in enumerate(trainloader):
            sample = torch.stack(sample)
            sample = sample.numpy().transpose(3, 4, 0, 1, 2).reshape(256, 306, -1)
            sample = cv2.resize(sample, (256, 256))
            sample = sample.reshape(256, 256, batchsize, 6, 3).transpose(2, 3, 4, 0, 1)
            sample = torch.Tensor(sample).to(device)
            t = []
            for i in range(len(target)):
                tmp = []
                for j in range(len(target[i]['bounding_box'])):
                    a, b, c, d = convert([800, 800], target[i]['bounding_box'][j].numpy())
                    e = int(target[i]['category'][j])
                    tmp.extend([e, a, b, c, d])
                t.append(convert_bbox2labels(tmp).transpose(2, 0, 1))
            labels = torch.Tensor(np.array(t)).float().to(device)
            pred = model(sample)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if it%500 == 0:
                print("Epoch %d/%d| Step %d/%d| Loss: %.2f" % (ep, epoch, it, len(labeled_trainset) // batchsize, loss))
            yl = yl + loss
        torch.save(model.state_dict(), "models_pkl/YOLOv1_epoch_model" + str(ep + 1) + ".pkl")
        torch.save(optimizer.state_dict(), "models_pkl/YOLOv1_epoch_optimizer" + str(ep + 1) + ".pkl")

        model.eval()
        yt = torch.Tensor([0]).cuda()
        with torch.no_grad():
            for it, (sample, target, road_image, extra) in enumerate(valloader):
                sample = torch.stack(sample)
                sample = sample.numpy().transpose(3, 4, 0, 1, 2).reshape(256, 306, -1)
                sample = cv2.resize(sample, (256, 256))
                sample = sample.reshape(256, 256, batchsize, 6, 3).transpose(2, 3, 4, 0, 1)
                sample = torch.Tensor(sample).to(device)
                t = []
                for i in range(len(target)):
                    tmp = []
                    for j in range(len(target[i]['bounding_box'])):
                        a, b, c, d = convert([800, 800], target[i]['bounding_box'][j].numpy())
                        e = int(target[i]['category'][j])
                        tmp.extend([e, a, b, c, d])
                    t.append(convert_bbox2labels(tmp).transpose(2, 0, 1))
                labels = torch.Tensor(np.array(t)).float().to(device)
                pred = model(sample)
                loss = criterion(pred, labels)*batchsize
                yt = yt + loss
            print("Epoch %d/%d|val Loss: %.2f" % (ep, epoch, yt/len(valloader)))


