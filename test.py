"""
from data import get_data_path, get_loader
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
num_classes = 19
input_size = (512,1024)
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
from torch.utils import data, model_zoo
from utils.helpers import colorize_mask
import utils.palette as palette
from model.deeplabv2 import Res_Deeplab
from torch.autograd import Variable


data_loader = get_loader('cityscapes')
data_path = get_data_path('cityscapes')
test_dataset = data_loader( data_path, img_size=input_size, img_mean = IMG_MEAN, is_transform=True, split='val')
testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
ignore_label = 250
import torchvision.transforms as transform

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])


            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('/home/s/kien/DACS_Visualization/CutMix/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('/home/s/kien/DACS_Visualization/CutMix/', str(epoch)+ id + '.png'))

model = Res_Deeplab(num_classes=num_classes)

checkpoint = torch.load("/home/s/kien/saved/DeepLabv2/06-20_16-04-UDA1_resume-06-21_10-14/checkpoint-iter110000.pth")
model.load_state_dict(checkpoint['model'])

model.cuda()
model.eval()

for index, batch in enumerate(testloader):
        image, label, size, name, _ = batch
        size = size[0]
        #if index > 500:
        #    break    model.eval()
        output  = model(Variable(image).cuda())
        output = interp(output)

        if index <= 20:
            # Saves two mixed images and the corresponding prediction
            save_image(image[0].cpu(),index,'input1',palette.CityScpates_palette)
            _, pred_u_s = torch.max(output, dim=1)
            save_image(pred_u_s[0].cpu(),index,'pred1',palette.CityScpates_palette)

import argparse
import json
import os
import os.path as osp
import numpy as np
from PIL import Image
from glob import glob


def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    pil_label = pil_label.resize((1280, 720), Image.NEAREST)
    label = np.asarray(pil_label)

    id_to_trainid = {
        7: 0,
        8: 1,
        11: 2,
        12: 3,
        13: 4,
        17: 5,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        31: 16,
        32: 17,
        33: 18
    }
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    sample_class_stats['file'] = file
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GTA annotations to TrainIds')
    parser.add_argument('gta_path', help='gta data path')
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    gta_path = args.gta_path
    out_dir = args.out_dir if args.out_dir else gta_path
    #os.mkdir(out_dir)

    #gt_dir = osp.join(gta_path, args.gt_dir)

    poly_files = []
    for mask_path in glob(gta_path + "/*/images/*.png"):
            #Error pictures inside GTA5 dataset
            mask_path = mask_path.replace("images", "labels")
            if(('/labels/15188' not in mask_path) and  ('/labels/17705' not in mask_path)):
                poly_files.append(mask_path)
    poly_files = sorted(poly_files)
    print(len(poly_files))
    
    sample_class_stats = []
    for file in poly_files:
        sample_class_stats.append(convert_to_train_id(file))
    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
from data.city_utils import get_rcs_class_probs
import numpy as np
import matplotlib.pyplot as plt
 
rcs_classes, rcs_classprob = get_rcs_class_probs(
                'data/samples', 0.01)
# creating the dataset
x = {}
trainid2name = {
            0:"Road",
            1:"S.walk",
            2:"Build.",
            3:"Wall",
            4:"Fence",
            5:"Pole",
            6:"Tr Light",
            7:"Tr. Sign",
            8:"Veget.",
            9:"Terrain",
            10:"Sky",
            11:"Person",
            12:"Rider",
            13:"Car",
            14:"Truck",
            15:"Bus",
            16:"Train",
            17:"M.cycle",
            18:"Bicycle"
        }
for i in range(19):
    x[trainid2name[rcs_classes[i]]] = rcs_classprob[i]
courses = list(x.keys())
values = list(x.values())
  
fig = plt.figure(figsize = (10, 6.2))
 
# creating the bar plot
plt.bar(courses, values, color ='blue',
        width = 0.4)
 
plt.xlabel("Class c", fontsize=20)
plt.ylabel("P(c)")
plt.xticks(rotation='vertical')
plt.title("Class sampling probability P(c) with T = 0.01")
plt.savefig("/home/s/kiendn/rcs_dacs_prototype/s.png")
plt.show()



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Res_Deeplab
from model.segformer import SegFormer

from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d

from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data import get_loader, get_data_path
from data.augmentations import *

import PIL
from torchvision import transforms

import time
import copy 
from core.configs import cfg
import os

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])

            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('/home/s/kiendn/DACS_Visualization/ClassMix/visualiseImagesI/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('/home/s/kiendn/DACS_Visualization/ClassMix/visualiseImagesL/', str(epoch)+ id + '.png'))

num_classes = 19

model = Res_Deeplab(num_classes=19)
weights_loaded = (torch.load("/home/s/kiendn/saved/DeepLabv2/best_model.pth", map_location = "cpu"))
model.load_state_dict(weights_loaded["model"], strict = False)

input_size = (512,1024)
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


data_loader = get_loader('cityscapes')
data_path = get_data_path('cityscapes')
test_dataset = data_loader( data_path, img_size=input_size, img_mean = IMG_MEAN, is_transform=True, split='val')


testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
ignore_label = 250




interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
mn = [20,201,202,207]
for i in (mn):
    img, lbl, img_path, lbl_path, img_name = test_dataset[i]
    print(lbl_path)
    
    img = img.unsqueeze(0)
    _,a = model(img)
    
    save_image(img[0].cpu(),j,'input',palette.CityScpates_palette)
    #save_image(inputs_u_s[1].cpu(),i_iter,'input2',palette.CityScpates_palette)
    _, pred_u_s = torch.max(interp(a), dim=1)
    #save_image(pred_u_s[0].cpu(),i_iter,'pred1',palette.CityScpates_palette)
    save_image(pred_u_s[0].cpu(),j,'pred',palette.CityScpates_palette)
    

model = Res_Deeplab(num_classes=19)
weights_loaded = (torch.load("/home/s/kiendn/saved/DeepLabv2/07-21_22-58-cut_classmix_proto_feat_out/best_model.pth", map_location = "cpu"))
model.load_state_dict(weights_loaded["model"], strict = False)

for i in range(10):
    j = 201+i
    img, lbl, img_path, lbl_path, img_name = test_dataset[j]
    img = img.unsqueeze(0)
    _,a = model(img)
    #save_image(inputs_u_s[1].cpu(),i_iter,'input2',palette.CityScpates_palette)
    _, pred_u_s = torch.max(interp(a), dim=1)
    #save_image(pred_u_s[0].cpu(),i_iter,'pred1',palette.CityScpates_palette)
    save_image(pred_u_s[0].cpu(),j,'pred_mclda',palette.CityScpates_palette)

"""


import argparse
import json
import os
import os.path as osp
import numpy as np
from PIL import Image
from glob import glob
import imageio


def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    label = np.asarray(imageio.v2.imread(file, format='PNG-FI'))[:,:,0]

    id_to_trainid = {
        3: 0,
        4: 1,
        2: 2,
        21: 3,
        5: 4,
        7: 5,
        15: 6,
        9: 7,
        6: 8,
        16: 9,  # not present in synthia
        1: 10,
        10: 11,
        17: 12,
        8: 13,
        18: 14,  # not present in synthia
        19: 15,
        20: 16,  # not present in synthia
        12: 17,
        11: 18
    }
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    sample_class_stats['file'] = file
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GTA annotations to TrainIds')
    parser.add_argument('gta_path', help='gta data path')
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    gta_path = args.gta_path
    out_dir = args.out_dir if args.out_dir else gta_path
    #os.mkdir(out_dir)

    #gt_dir = osp.join(gta_path, args.gt_dir)

    poly_files = []
    for mask_path in glob(gta_path + "/*.png"):
            #Error pictures inside GTA5 dataset
            mask_path = mask_path.replace("images", "labels")
            poly_files.append(mask_path)
    poly_files = sorted(poly_files)
    print(len(poly_files))
    
    sample_class_stats = []
    ii = 0
    for file in poly_files:
        ii = ii + 1
        if ii % 200 == 0:
            print(ii)
        sample_class_stats.append(convert_to_train_id(file))
    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
