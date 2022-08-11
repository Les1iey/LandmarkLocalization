import os
import math
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LandmarkDataset(Dataset):
    def __init__(self, img_dir, gt_dir, height, width, num_classes, sigma,alpha):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.height = int(height)
        self.width = int(width)
        self.image_size = [self.height,self.width]
        self.num_classes = num_classes
        self.sigma = sigma
        self.alpha = alpha
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        #get image
        img_name = self.img_names[idx]
        img_file = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_file)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        #img = img / 255
        transform = transforms.Compose([
            #transforms.Normalize([121.78, 121.78, 121.78], [74.36, 74.36, 74.36])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        )
        img = transform(img)

        # generate ground-truth
        gt_file = self.gt_dir + '/' + img_name.split('.')[0] + '.txt'
        gt_array = np.loadtxt(gt_file, delimiter=",")


        heatmaps = np.zeros([self.num_classes, self.height, self.width])
        for i in range(self.num_classes):
            landmark_coordinate = gt_array[i]
            heatmaps[i] = generate_heatmap(self.image_size,landmark_coordinate,self.sigma,self.alpha)
        heatmaps = torch.from_numpy(heatmaps).float()


        return img, heatmaps,img_name



def generate_heatmap(image_size,coords,sigma,alpha, dtype=np.float32):

    heatmap = np.zeros(image_size, dtype=dtype)
    size_sigma_factor = 10

    # flip point from [x, y, z] to [z, y, x]
    flipped_coords = np.flip(coords, 0)
    region_start = (flipped_coords - sigma * size_sigma_factor / 2).astype(int)
    region_end = (flipped_coords + sigma * size_sigma_factor / 2).astype(int)
    region_start = np.maximum(0, region_start).astype(int)
    region_end = np.minimum(image_size, region_end).astype(int)

    # return zero landmark, if region is invalid, i.e., landmark is outside of image
    if np.any(region_start >= region_end):
        return heatmap

    region_size = (region_end - region_start).astype(int)

    dy, dx = np.meshgrid(range(region_size[1]), range(region_size[0]))
    x_diff = dx + region_start[0] - flipped_coords[0]
    y_diff = dy + region_start[1] - flipped_coords[1]

    squared_distances = x_diff * x_diff + y_diff * y_diff

    cropped_heatmap = np.exp(-squared_distances / (2 * math.pow(sigma, 2)))

    heatmap[region_start[0]:region_end[0],
    region_start[1]:region_end[1]] = cropped_heatmap[:, :]

    #
    heatmap = np.power(alpha, heatmap)
    heatmap[heatmap <= 1.05] = 0

    return heatmap