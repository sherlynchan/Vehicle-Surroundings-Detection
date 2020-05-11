"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import detector
import road_map
import cv2
import numpy as np
from skimage.measure import label, regionprops

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'Onlooker'
    round_number = 1
    team_member = ['Shaoling Chen', 'Yunan Hu', 'Shizhan Gong']
    contact_email = '@nyu.edu'

    def __init__(self, model_file1='/scratch/sc6995/mujupyter/dl/DS-GA-1008-deep-learning/models_pkl/pretrained_bbox_segment_epoch15.pkl', model_file2='/home/sg5722/models_pkl/road_map_model_epoch8.pkl'):
        # You should
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        #
        self.model_detector = detector.res_detector()
        self.model_map = road_map.res_roadmap()
        self.model_detector.cuda()
        self.model_map.cuda()
        self.model_detector.load_state_dict(torch.load(model_file1))
        self.model_map.load_state_dict(torch.load(model_file2))

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        batch_size = len(samples)
        samples = samples.cpu().numpy().transpose(3, 4, 0, 1, 2).reshape(256, 306, -1)
        samples = cv2.resize(samples, (256, 256))
        samples = samples.reshape(256, 256, -1, 6, 3).transpose(2, 3, 4, 0, 1)
        samples = torch.Tensor(samples).cuda()
        with torch.no_grad():
            out = self.model_detector(samples)
            pred_bi = ((out > 0.5) + 0)
        bbox = []
        for i in range(batch_size):
            props = regionprops(label(1 - pred_bi[i].reshape(800, 800).cpu()))
            single_bbox = []
            for prop in props:
                x0, y0 = ((prop.bbox[1] - 400) / 10), (-(prop.bbox[0] - 400) / 10)
                x1, y1 = ((prop.bbox[3] - 400) / 10), (-(prop.bbox[2] - 400) / 10)
                tmp = []
                tmp.append([x0, y0])
                tmp.append([x0, y1])
                tmp.append([x1, y0])
                tmp.append([x1, y1])
                single_bbox.append(list(np.array(tmp).T))
            if len(single_bbox) == 0:
                single_bbox.append(list(np.random.rand(2, 4) * 10))
            bbox.append(torch.Tensor(np.array(single_bbox)).cuda())

        return tuple(bbox)

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        samples = samples.cpu().numpy().transpose(3, 4, 0, 1, 2).reshape(256, 306, -1)
        samples = cv2.resize(samples, (256, 256))
        samples = samples.reshape(256, 256, -1, 6, 3).transpose(2, 3, 4, 0, 1)
        samples = torch.Tensor(samples).cuda()
        with torch.no_grad():
            out = self.model_map(samples)
        return out.view(-1,800,800) > 0.5
