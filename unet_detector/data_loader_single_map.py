import os
from PIL import Image

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from helper import convert_map_to_lane_map, convert_map_to_road_map

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
]


# The dataset class for labeled data.
class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_file, scene_index, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """
        self.type = type
        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            images.append(self.transform(image))
        image_tensor = torch.stack(images)

        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y', 'bl_y', 'br_y']].to_numpy()
        # categories = data_entries.category_id.to_numpy()

        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)
        road_image_ = road_image.numpy().astype('uint8')
        road_image_tensor = torch.stack(self.convert_to_single_map(road_image_, 0))

        corners = torch.as_tensor(corners).view(-1, 2, 4)
        box = self.convert_label_to_mask(corners)
        box_tensor = torch.stack(self.convert_to_single_map(box, 1))

        return image_tensor, box, box_tensor, road_image, road_image_tensor

    def convert_label_to_mask(self, labels):
        box = []
        for corners in labels:
            corners[0, :] = corners[0, :] * 10 + 400
            corners[1, :] = -corners[1, :] * 10 + 400
            point_sequence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])
            point_sequence = np.array(point_sequence, 'int32')
            box.append(point_sequence)
        img = np.ones((800, 800), np.uint8)
        img = cv2.fillPoly(img, box, 0)
        return img

    def convert_to_single_map(self, img, c=1):
        imgs = []
        # fl
        img_fl = img.copy()[0:400, 400:800]
        tri_fl = np.array([[400, 200], [400, 400], [0, 400], [400, 200]], 'int32')
        img_fl = cv2.fillConvexPoly(img_fl, tri_fl, c)
        imgs.append(self.transform(img_fl))
        # f
        img_f = img.copy()[200:600, 400:800]
        tri1_f = np.array([[0, 0], [400, 0], [0, 200]], 'int32')
        tri2_f = np.array([[0, 200], [0, 400], [400, 400]], 'int32')
        cv2.fillPoly(img_f, [tri1_f], c)
        cv2.fillPoly(img_f, [tri2_f], c)
        imgs.append(self.transform(img_f))
        # fr
        img_fr = img.copy()[400:800, 400:800]
        tri_fr = np.array([[400, 200], [0, 0], [400, 0]], 'int32')
        img_fr = cv2.fillPoly(img_fr, [tri_fr], c)
        imgs.append(self.transform(img_fr))
        # bl
        img_bl = img.copy()[0:400, 0:400]
        tri_bl = np.array([[0, 200], [0, 400], [400, 400], [0, 200]], 'int32')
        img_bl = cv2.fillPoly(img_bl, [tri_bl], c)
        imgs.append(self.transform(img_bl))
        # b
        img_b = img.copy()[200:600, 0:400]
        tri1_b = np.array([[0, 0], [400, 0], [400, 200], [0, 0]], 'int32')
        tri2_b = np.array([[0, 400], [400, 200], [400, 400], [0, 400]], 'int32')
        img_b = cv2.fillPoly(img_b, [tri1_b, tri2_b], c)
        imgs.append(self.transform(img_b))
        # br
        img_br = img.copy()[400:800, 0:400]
        tri_br = np.array([[0, 0], [400, 0], [0, 200], [0, 0]], 'int32')
        img_br = cv2.fillPoly(img_br, [tri_br], c)
        imgs.append(self.transform(img_br))
        return imgs