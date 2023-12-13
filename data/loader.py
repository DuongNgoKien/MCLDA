import os
import torch
import numpy as np
from skimage.io import imread
from torch.utils import data

class loader(data.Dataset):

    def __init__(
        self,
        root,
        is_transform=False,
        img_mean = np.array([73.15835921, 82.90891754, 72.39239876])
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.is_transform = is_transform
        self.mean = img_mean

        self.files = os.listdir(root)
        

    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[index].rstrip()
        img_path = os.path.join(self.root, img_path)

        img = imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if self.is_transform:
            img = self.transform(img)

        return img, img_path

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img