import numpy as np
import pandas as pd
from skimage import transform
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class forensics_datareader(Dataset):
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
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if mask_path == 'None':
            mask = np.zeros(img.shape,np.uint8)
        else:
            mask = cv2.imread(mask_path, 1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        label = self.data_path_details.iloc[idx, 2]
        if label != 0:
            label = 1
        samples = {'image': img,
                  'mask': mask,
                   'label': label,
                   'mask_path': mask_path}
        if self.transform:
            samples = self.transform(samples)
        return samples

class forensics_transforms(object):
    def __init__(self, output_size=512, mask_size=512, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

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
        label = sample['label']
        mask_path = sample['mask_path']

        image = transform.resize(image, self.output_size)
        mask = transform.resize(mask, self.mask_size)
        if len(mask.shape) == 3:
            mask = np.mean(mask, axis=2)

        mask = (mask > 0.5).astype(int)
        
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).long()
        label = torch.tensor(label).long()

        sample_trans = {'images': image,
                  'masks': mask,
                   'labels': label,
                   'mask_paths': mask_path}
        return sample_trans