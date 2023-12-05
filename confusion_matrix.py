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

#Global configs
NUM_CLASSES = 19
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
MODEL = 'deeplabv2' # deeeplabv2, deeplabv3p
device = "cuda"

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion Matrix',
                          cmap=None,
                          normalize=True,
                          name_of_plot='mycm_111.png'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Greens')

    plt.figure(figsize=(30, 24))
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=23)
        plt.yticks(tick_marks, target_names, fontsize=23)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title, fontsize=23)
    # plt.colorbar(shrink=1, labelsize=20)
    cbar = plt.colorbar(shrink=1)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(23)
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",fontsize=20,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",fontsize=20,
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True labels', fontsize=23)
    plt.xlabel('Predicted labels', fontsize=23)
    print(f">?>>>{name_of_plot}")
    plt.savefig(name_of_plot, bbox_inches = "tight")
    plt.close()

def get_iou(data_list, class_num, dataset, save_path=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)
        
    aveJ, j_list, M = ConfM.jaccard()


    classes = np.array(("road", "sidewalk",
        "building", "wall", "fence", "pole",
        "traffic_light", "traffic_sign", "vegetation",
        "terrain", "sky", "person", "rider",
        "car", "truck", "bus",
        "train", "motorcycle", "bicycle"))
    
    if save_path:
        plot_confusion_matrix(M, target_names=\
            ["road","sidewalk","building","wall",
            "fence","pole","light","sign","vegetation",
            "terrain","sky","person","rider","car",
            "truck","bus","train","motocycle","bicycle"],\
            name_of_plot = save_path)
        
    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]))

    print('meanIOU: ' + str(aveJ) + '\n')

    return ConfM


#This version use CPU to evaluate
def evaluate(model, dataset, ignore_label=250, save_output_images=False, save_dir=None, input_size=(512,1024), saved_name = "Confusion_Matrix.png"):
    
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

    data_list = []
    colorize = VOCColorize()

    total_loss = []

    for index, batch in enumerate(testloader):
        image, label, size, name, _ = batch
        size = size[0]
        #if index > 500:
        #    break
        with torch.no_grad():
            output  = model(Variable(image).to(device))[1]
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

            data_list.append([gt.flatten(), output.flatten()])

        if (index+1) % 100 == 0:
            print('%d processed'%(index+1))

    if save_dir:
        filename = os.path.join(save_dir, saved_name)
    else:
        filename = None
    mIoU = get_iou(data_list, num_classes, dataset, filename)
    loss = np.mean(total_loss)
    return mIoU, loss




def evaluate_experiment_version():
    #global_name = f"CM_{exp}_{ver}.png"

    model = Res_Deeplab(19)
    checkpoint = torch.load("/home/s/kiendn/saved/DeepLabv2/07-21_22-58-cut_classmix_proto_feat_out/best_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    model = model.to(device)
    evaluate(model, dataset="cityscapes", ignore_label=250, save_output_images=True, save_dir="/home/s/kiendn/saved/DeepLabv2/07-21_22-58-cut_classmix_proto_feat_out/", input_size=(512, 1024), saved_name='Confusion_Matrix')

# Define experiments and versions
"""Experiments = ["DACS_FFT"]

# Create a list to store the threads
threads = []

# Iterate over experiments and versions
for exp in Experiments:
    versions = glob(f"/home/admin_mcn/kien/saved/DeepLabv2/{exp}/*.pth")
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
"""
evaluate_experiment_version()
