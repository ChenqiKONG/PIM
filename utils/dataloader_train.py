import numpy as np
import argparse
import os

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from PIL import Image
import random
import cv2

class forensics_datareader_train(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_path_details = pd.read_csv(csv_file,header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data_path_details)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.data_path_details.iloc[idx, 0]
        mask_path = self.data_path_details.iloc[idx, 1]
        boundary_path = self.data_path_details.iloc[idx, 2]
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if mask_path == 'None':
            mask = np.zeros(img.shape,np.uint8)
            boundary = np.zeros(img.shape,np.uint8)
        else:
            mask = cv2.imread(mask_path, 1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            boundary = cv2.imread(boundary_path, 1)
            boundary = cv2.cvtColor(boundary, cv2.COLOR_BGR2RGB)

        label = self.data_path_details.iloc[idx, 3]
        if label != 0:
            label = 1
        samples = {'image': img,
                  'mask': mask,
                  'boundary': boundary, 
                   'label': label}
        if self.transform:
            samples = self.transform(samples)
        return samples

class forensics_transforms_train(object):
    def __init__(self, output_size=512, mask_size=128, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
            self.mask_size = (mask_size, mask_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            self.mask_size = mask_size

        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        boundary = sample['boundary']
        label = sample['label']

        image = transform.resize(image, self.output_size)
        img_label = transform.resize(image, (128, 128))
        mask = transform.resize(mask, self.mask_size)
        boundary = transform.resize(boundary, self.mask_size)
        if len(mask.shape) == 3:
            mask = np.mean(mask, axis=2)
        if len(boundary.shape) == 3:
            boundary = np.mean(boundary, axis=2)

        mask = (mask > 0.5).astype(int)
        boundary = (boundary > 0.5).astype(int)
        
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)

        img_label = (img_label - self.mean) / self.std
        img_label = img_label.transpose(2, 0, 1)

        image = torch.from_numpy(image.copy()).float()
        img_label = torch.from_numpy(img_label.copy()).float()
        mask = torch.from_numpy(mask.copy()).long()
        boundary = torch.from_numpy(boundary.copy()).long()
        label = torch.tensor(label).long()
        
        # print(image.size(), mask.size())
        # print(torch.max(image), torch.min(image), torch.max(mask), torch.min(mask))

        sample_trans = {'images': image,
                  'masks': mask,
                  'boundaries': boundary,
                   'labels': label,
                   'img_labels': img_label}
        return sample_trans