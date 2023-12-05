import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
import numpy as np
import os


from model.deeplabv2 import Res_Deeplab
from data import get_data_path, get_loader
from utils.loss import CrossEntropy2d
from evaluateUDA import VOCColorize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from glob import glob
import threading
import cv2

#Global configs
NUM_CLASSES = 19
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
MODEL = 'deeplabv2' # deeeplabv2, deeplabv3p
device = "cuda"


ID_TO_COLOR = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

#This version use CPU to evaluate
def evaluate(model, dataset, ignore_label=250, save_output_images=False, save_dir=None, input_size=(512,1024), saved_name = "Confusion_Matrix.png"):
    H, W = input_size
    
    if save_dir:
        filename = os.path.join(save_dir, saved_name)
    else:
        filename = None

    if not os.path.exists(filename):
        os.mkdir(filename)
    
    if dataset == 'cityscapes':
        num_classes = 19
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        test_dataset = data_loader( data_path, img_size=input_size, img_mean = IMG_MEAN, is_transform=True, split='val')
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False)
        interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
        ignore_label = 250

    elif dataset == 'gta':
        num_classes = 19
        data_loader = get_loader('gta')
        data_path = get_data_path('gta')
        test_dataset = data_loader(data_path, list_path = './data/gta5_list/train.txt', img_size=(1280,720), mean=IMG_MEAN)
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=False)
        interp = nn.Upsample(size=(720,1280), mode='bilinear', align_corners=True)
        ignore_label = 255

    print('Evaluating, found ' + str(len(testloader)) + ' images.')

    colorize = VOCColorize()

    total_loss = []

    for index, batch in enumerate(testloader):
        print(index)
        image, label, size, name, _ = batch
        size = size[0]
        if index > 20: #Only draw the first 20 pictures 
           break
        with torch.no_grad():
            output  = model(Variable(image).to(device))
            output = interp(output)

            label_cuda = Variable(label.long()).to(device)
            criterion = CrossEntropy2d(ignore_label=ignore_label).to(device)  # Ignore label ??
            loss = criterion(output, label_cuda)
            total_loss.append(loss.item())

            output = output.cpu().data[0].numpy()

            if dataset == 'cityscapes':
                gt = np.asarray(label[0].numpy(), dtype=np.int32)
            elif dataset == 'gta':
                gt = np.asarray(label[0].numpy(), dtype=np.int32)

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int32)
            
            
            output = np.uint8(output)
            gt = np.uint8(gt)
            prediction_mask = np.uint8(np.zeros((H,W,3)))
            true_mask = np.uint8(np.zeros((H,W,3)))
            for i in range(NUM_CLASSES):
                prediction_mask[output==i] = ID_TO_COLOR[i]
                true_mask[gt ==i] = ID_TO_COLOR[i]
            ori_img = cv2.imread(name[0].replace("_gtFine_labelIds", "_leftImg8bit").replace("/gtFine/", "/leftImg8bit/"), cv2.IMREAD_COLOR)
            ori_img = cv2.resize(ori_img, dsize=(1024,512), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{filename}/{index}.jpg",\
                cv2.cvtColor(np.concatenate([ori_img, prediction_mask, true_mask], axis=1), cv2.COLOR_RGB2BGR))


        if (index+1) % 100 == 0:
            print('%d processed'%(index+1))

    
    return None




def evaluate_experiment_version(exp, ver):
    global_name = f"Visual_{exp}_{ver}"

    model = Res_Deeplab(19)
    checkpoint = torch.load(f"/home/s/nvanh/saved/DeepLabv2/{exp}/{ver}.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    model = model.to(device)
    evaluate(model, dataset="cityscapes", ignore_label=255, save_output_images=True, save_dir="/mnt/nvanh/Visualization", input_size=(512, 1024), saved_name = global_name)

# Define experiments and versions
Experiments = ["Original_DACS"]

# Create a list to store the threads
threads = []

# Iterate over experiments and versions
for exp in Experiments:
    versions = glob(f"/home/s/nvanh/saved/DeepLabv2/{exp}/*.pth")
    versions = [item.split("/")[-1].replace(".pth", "") for item in versions]

    for ver in versions:
        # Create a thread for each experiment and version
        t = threading.Thread(target=evaluate_experiment_version, args=(exp, ver))
        threads.append(t)

# Start the threads
for t in threads:
    t.start()

# Wait for all threads to finish
for t in threads:
    t.join()

