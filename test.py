import os
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data_helper import LabeledDataset
from helper import compute_ats_bounding_boxes, compute_ts_road_map
from models import ResNetVAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

opt_testset = False
image_folder = 'data'
annotation_csv = 'data/annotation.csv'
transform = torchvision.transforms.ToTensor()
if opt_testset:
    labeled_scene_index = np.arange(134, 148)
else:
    labeled_scene_index = np.arange(130, 134)


# For road map task
labeled_trainset_task2 = LabeledDataset(
    image_folder=image_folder,
    annotation_file=annotation_csv,
    scene_index=labeled_scene_index,
    transform=transform,
    extra_info=False
    )
dataloader_task2 = torch.utils.data.DataLoader(
    labeled_trainset_task2,
    batch_size=1,
    shuffle=False,
    num_workers=4
    )

model = ResNetVAE().to(device)
model_fn = 'models_pkl/ResNetAE_0507_yhu.pkl'
model.load_state_dict(torch.load(model_fn, map_location=device))

total = 0
total_ats_bounding_boxes = 0
total_ts_road_map = 0
with torch.no_grad():

    for i, data in enumerate(dataloader_task2):
        total += 1
        sample, target, road_image = data
        sample = sample.to(device)

        predicted_road_map = (model(sample)[0] > 0.5).detach().cpu()
        ts_road_map = compute_ts_road_map(predicted_road_map, road_image)
        total_ts_road_map += ts_road_map

print(f'Road Map Score: {total_ts_road_map / total:.4}')