import os
import os.path as osp
import numpy as np
import random
# import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from glob import glob
import json
from data.city_utils import get_rcs_class_probs

class GTA5DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, augmentations = None, img_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=250, load_full = None):
        self.root = root
        # self.list_path = list_path
        self.img_size = img_size
        self.list_path = list_path
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.augmentations = augmentations
        self.load_full = load_full
        self.img_ids = [i_id.strip().replace(".png", "") for i_id in open(list_path)]
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        # self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        
        self.rcs_class_temp = 0.01
        self.rcs_min_pixels = 3000
        self.rcs_min_crop_ratio = 0.5
        self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                'data/gta5_list', self.rcs_class_temp)
        self.file_to_idx = {}
        for img_path in glob(self.root + "/*/images/*.png"):
            mask_path = img_path.replace("images", "labels")
            name = img_path.split("/")[-1].replace(".png", "")
            #Error pictures inside GTA5 dataset
            if(('/images/15188' not in img_path) and  ('/images/17705' not in img_path) and name in self.img_ids):
                self.files.append(
                    {
                        "img": img_path,
                        "label": mask_path,
                        "name": name
                    }
                )
                self.file_to_idx[name] = {
                        "img": img_path,
                        "label": mask_path,
                        "name": name
                }
            else:
                print(img_path, mask_path)
        with open('data/gta5_list/samples_with_class.json', 'r') as of:
                samples_with_class_and_n = json.load(of)
        samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
        }
        self.samples_with_class = {}
        for c in self.rcs_classes:
            self.samples_with_class[c] = []
            for file, pixels in samples_with_class_and_n[c]:
                if pixels > self.rcs_min_pixels:
                    self.samples_with_class[c].append(file.split('/')[-1].replace('.png', ''))
            assert len(self.samples_with_class[c]) > 0
        

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        if self.load_full:
            datafiles = self.files[index]
        else:
            c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
            selected_file = np.random.choice(self.samples_with_class[c])
            datafiles = self.file_to_idx[selected_file]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        
        name = datafiles["name"]

        # resize
        image = image.resize(self.img_size, Image.BICUBIC)
        label = label.resize(self.img_size, Image.NEAREST)

        image = np.asarray(image, np.uint8)
        label = np.asarray(label, np.uint8)

        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        if self.load_full:
            image_c = np.asarray(image, np.float32)
            label_c = np.asarray(label_copy, np.float32)

        if self.augmentations is not None:
            for _ in range(20):
                image_tmp, label_tmp = self.augmentations(image, label_copy)
                image_tmp = np.asarray(image_tmp, np.float32)
                label_tmp = np.asarray(label_tmp, np.float32)
                # re-assign labels to match the format of Cityscapes
                n_class = np.sum(label_tmp == c)
                if n_class > 1500:
                    break
            image_c = image_tmp
            label_c = label_tmp
            
        image_c = image_c[:, :, ::-1]  # change to BGR
        image_c -= self.mean
        image_c = image_c.transpose((2, 0, 1))
        if self.load_full:
            return image_c.copy(), label_c.copy()
        else:
            return image_c.copy(), label_c.copy(), name, c


# if __name__ == '__main__':
#     dst = GTA5DataSet(root = "/home/s/taint/dataset/gta5")
#     trainloader = data.DataLoader(dst, batch_size=4)
#     for i, data in enumerate(trainloader):
#         imgs, labels,_,_ = data
#         # img = torchvision.utils.make_grid(imgs).numpy()
#         # img = np.transpose(img, (1, 2, 0))
#         # img = img[:, :, ::-1]
#         print(imgs.shape)
