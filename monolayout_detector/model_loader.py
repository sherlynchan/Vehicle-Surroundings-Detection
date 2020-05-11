"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
import model
from skimage.measure import label, regionprops
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [3, 3]
matplotlib.rcParams['figure.dpi'] = 200
model_path_1 = "../models/os_128_0/weights_90/"
model_path_2 = "../models/os_128_1/weights_50/"
model_path_3 = "../models/os_128_2/weights_45/"
model_path_4 = "../models/os_128_3/weights_65/"
model_path_5 = "../models/os_128_4/weights_95/"
model_path_6 = "../models/os_128_5/weights_65/"
# model_path_1 = "../models/os_64_0/weights_95/"
# model_path_2 = "../models/os64_1/weights_95/"
# model_path_3 = "../models/os_64_2/weights_45/"
# model_path_4 = "../models/os_64_3/weights_90/"
# model_path_5 = "../models/os_64_4/weights_15/"
# model_path_6 = "../models/os_64_5/weights_95/"
model_path = [model_path_1,model_path_2,model_path_3,model_path_4,model_path_5,model_path_6]
dic = {0: (400, 0), 1: (400, 200), 2: (400, 400), 3: (0, 0), 4: (0, 200), 5: (0, 400)}
device = torch.device("cuda")
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
    team_number = 44
    round_number = 2
    team_member = ['Shaoling Chen', 'Yunan Hu', 'Shizhan Gong']
    contact_email = 'sc6995@nyu.edu'

    def __init__(self, model_file1=model_path):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        #
        self.models_detector = []
        for i in range(6):
            models = {}
            encoder_path = os.path.join(model_file1[i], "encoder.pth")
            encoder_dict = torch.load(encoder_path, map_location=device)
            feed_height = encoder_dict["height"]
            feed_width = encoder_dict["width"]
            models["encoder"] = model.Encoder(feed_width, feed_height, False)
            filtered_dict_enc = {k: v for k, v in encoder_dict.items() if k in models["encoder"].state_dict()}
            models["encoder"].load_state_dict(filtered_dict_enc)
            decoder_path = os.path.join(model_file1[i], "decoder.pth")
            models["decoder"] = model.Decoder(models["encoder"].num_ch_enc)
            models["decoder"].load_state_dict(torch.load(decoder_path, map_location=device))
            for key in models.keys():
                models[key].to(device)
                models[key].eval()
            self.models_detector.append(models)


    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        bbox_sample = []
        samples = samples.cpu().numpy().transpose(3, 4, 0, 1, 2).reshape(256, 306, -1)
        samples = cv2.resize(samples, (512, 512))
        samples = samples.reshape(512, 512, -1, 6, 3).transpose(2, 3, 4, 0, 1)
        samples = torch.Tensor(samples).cuda()
        # transform each sample
        bboxs = []
        for i in range(6):
            with torch.no_grad():
                models = self.models_detector[i]
                input_image = samples[:, i, :]
                input_image = input_image.to(device)
                features = models["encoder"](input_image)
                tv = models["decoder"](features, is_training=False)
                tv = nn.functional.upsample(tv, size=400, mode='bilinear', align_corners=False)
                tv_np = tv.squeeze().cpu().numpy()
                true_top_view = np.zeros((tv_np.shape[1], tv_np.shape[2]))
                true_top_view[tv_np[1] > tv_np[0]] = 255
                props = regionprops(label(true_top_view))
                if props:
                    for prop in props:
                        # x0, y0 = prop.bbox[1] + dic[i][0], prop.bbox[0] + dic[i][1]
                        # x1, y1 = prop.bbox[3] + dic[i][0], prop.bbox[2] + dic[i][1]
                        x0, y0 = ((prop.bbox[1] + dic[i][0] - 400)/10), (-(prop.bbox[0] + dic[i][1]-400)/10)
                        x1, y1 = ((prop.bbox[3] + dic[i][0] - 400)/10), (-(prop.bbox[2] + dic[i][1]-400)/10)
                        bboxs.append([[x1, x1, x0, x0], [y0, y1, y0, y1]])
        bbox_sample.append(torch.Tensor(np.array(bboxs)).double().cuda())
        return bbox_sample
        #return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        
        return torch.rand(1, 800, 800) > 0.5