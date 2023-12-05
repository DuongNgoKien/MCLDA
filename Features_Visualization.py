import torch
from model.deeplabv2 import Bottleneck, affine_par
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F
# from torchinfo import summary
from torch import nn
from data import get_data_path, get_loader
from torch.utils import data
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


np.random.seed(1)
ID_TO_COLOR = [ 
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
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
trainid2name = {
            0:"road",
            1:"sidewalk",
            2:"building",
            3:"wall",
            4:"fence",
            5:"pole",
            6:"light",
            7:"sign",
            8:"vegetation",
            9:"terrain",
            10:"sky",
            11:"person",
            12:"rider",
            13:"car",
            14:"truck",
            15:"bus",
            16:"train",
            17:"motorcycle",
            18:"bicycle"
        }

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        # self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    # def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
    #     return block(dilation_series,padding_series,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)

        return x

def Res_Deeplab(num_classes):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model
    
model = Res_Deeplab(num_classes=19)
weights_loaded = (torch.load("/home/s/kiendn/saved/DeepLabv2/best_model.pth", map_location = "cpu"))
model.load_state_dict(weights_loaded["model"], strict = False)
# summary(model, input_size = (1,3,512,1024))

#Output as 1x2048x65x129




# Cityscapes validation
num_classes = 19
## For Cityscapes
data_loader = get_loader('cityscapes')
data_path = get_data_path('cityscapes')
test_dataset = data_loader( data_path, img_size=(512,1024), img_mean = IMG_MEAN, is_transform=True, split='val')
testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

## For GTA5
#data_loader = get_loader('cityscapes')
#data_path = get_data_path('cityscapes')
#train_dataset = data_loader(data_path, is_transform=True, augmentations=None, img_size=(512, 1024), img_mean = IMG_MEAN)
#testloader = data.DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=True)

# Set the model to evaluation mode
model.eval()
model = model.cuda()

# Extract features from the Segformer model
features = []
labels = []
number_pixels_keep_each_class = 1000
count = 0
with torch.no_grad():
    # for image, label, _, _, _ in tqdm(testloader):
    for image, label, _, _, _ in tqdm(testloader): 
        image = image.cuda()
        label = label.cuda()
        output = model(image)  # Adjust input shape as per model's requirement
        output = output.squeeze(0)
        C,H,W = output.shape
        
        # ## Only for visualization
        # output = output.detach().cpu().numpy()
        # list_feature_flatten = np.argmax(output, axis =0)
        # list_feature_flatten = cv2.resize(list_feature_flatten, (1024, 512), interpolation= cv2.INTER_NEAREST)
        # label_cpu = torch.squeeze(label, dim =0).detach().cpu().numpy()
        # label_cpu = np.uint8(label_cpu)
        # visual_features = np.zeros((512,1024,3))
        # visual_label = np.zeros((512,1024,3))
        # for i in range(19):
        #     visual_features[list_feature_flatten == i] = ID_TO_COLOR[i]
        
        # for i in range(19):
        #     visual_label[label_cpu == i] = ID_TO_COLOR[i]
            
        # visual = np.concatenate([visual_features, visual_label], axis = 1)
        # visual = np.float32(visual)
        # cv2.imwrite("/home/admin_mcn/nvanh/DACS/test.png", cv2.cvtColor(visual, cv2.COLOR_RGB2BGR))
        # break
        # #End visualization
        output = output.reshape(C, H*W)
        list_feature_flatten = output.transpose(1,0).cpu().numpy()
        
        #Downscale label to be 4x smaller
        label = torch.nn.functional.interpolate(label.unsqueeze(0).float(), size=(H, W), mode='nearest').squeeze(0)
        label_cpu = torch.squeeze(label, dim =0).detach().cpu().numpy()
        label_cpu = np.uint8(label_cpu)
        list_label_flatten = label_cpu.flatten()
        
        
    
        
        #Remove all the ignore label
        idx = (list_label_flatten == 250)
        list_label_flatten = np.delete(list_label_flatten, idx)
        list_feature_flatten = np.delete(list_feature_flatten, idx, axis = 0)
        
        
        
        
        #Chooses randomly 100 pixels from each class
        keep_indices = []
        for i in range(19):
            idx = np.where(list_label_flatten == i)[0]
            if len(idx) <= number_pixels_keep_each_class:
                # continue
                keep_indices.extend(list(np.random.choice(idx, size=len(idx), replace=False)))
            else:
                keep_indices.extend(list(np.random.choice(idx, size=number_pixels_keep_each_class, replace=False)))
        
        keep_indices = sorted(keep_indices)

        list_label_flatten = list_label_flatten[keep_indices]
        list_feature_flatten = list_feature_flatten[keep_indices]

        # #Save the features along with the label
        # features.append(list_feature_flatten)
        # labels.append(list_label_flatten)
        if count == 0:
            features = list_feature_flatten
            labels = list_label_flatten
        else:
            features = np.concatenate([features, list_feature_flatten], axis = 0)
            labels = np.concatenate([labels, list_label_flatten], axis = 0)
        del list_feature_flatten
        del list_label_flatten
        count +=1
        #print(features.shape, labels.shape)
        if count > 200:
            break
        

# features = np.concatenate(features, axis = 0)
# labels = np.concatenate(labels, axis =0)

number_pixels_plot = 10000
CityScapes_palette = ['#804080','#F423E8','#464646','#66669C','#BE9999','#999999',
                        '#FAAA1E','#DCDC00','#6B8E23','#98FB98','#4682B4','#DC143C','#FA0000','#00008E',
                        '#000046','#003C64','#000000','#0000E6','#770B20']

#Chooses randomly 5000 pixels from each class
keep_indices = []
for i in range(19):
    idx = np.where(labels == i)[0]
    if len(idx) <= number_pixels_plot:
        # continue
        keep_indices.extend(list(np.random.choice(idx, size=len(idx), replace=False)))
    else:
        keep_indices.extend(list(np.random.choice(idx, size=number_pixels_plot, replace=False)))

keep_indices = sorted(keep_indices)

features = features[keep_indices]
labels = labels[keep_indices]

print(f"The features shape is {features.shape}, The labels shape is {labels.shape}")
print(f"The labels inside this :{np.unique(labels)}")
reducer = umap.UMAP(random_state=42, n_neighbors=1000, min_dist=0.1)
embeddings = reducer.fit_transform(features)

# Plot the embeddings
df = pd.DataFrame()
df["y"] = labels
df["Emb1"] = embeddings[:,0]
df["Emb2"] = embeddings[:,1]
#df.to_csv("saved_df.csv")


list_color = sns.color_palette(CityScapes_palette, 19)
palette = {trainid2name[i]:list_color[i] for i in range(19)}
sns.scatterplot(x="Emb1", y="Emb2", hue= [trainid2name[item] for item in df.y.tolist()],
                palette=palette,
                marker="s",  # Use square markers
                s=1,  # Size of markers
                data=df).set(title="Cityscapes features map")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.savefig("/home/s/kiendn/saved/DeepLabv2/DeepLabv2_Feature_Visualization_all_classes.png", bbox_inches="tight", dpi=300)

plt.close()

"""


#Chooses randomly 5000 pixels from each class
keep_indices = []
for i in [0, 1, 11, 12, 15, 16, 17, 18]:
    idx = np.where(labels == i)[0]
    if len(idx) <= number_pixels_plot:
        # continue
        keep_indices.extend(list(np.random.choice(idx, size=len(idx), replace=False)))
    else:
        keep_indices.extend(list(np.random.choice(idx, size=number_pixels_plot, replace=False)))

keep_indices = sorted(keep_indices)

features = features[keep_indices]
labels = labels[keep_indices]


print(f"The features shape is {features.shape}, The labels shape is {labels.shape}")
print(f"The labels inside this :{np.unique(labels)}")


reducer = umap.UMAP(random_state=42, n_neighbors=1000, min_dist=0.1)
embeddings = reducer.fit_transform(features)

# Plot the embeddings
df = pd.DataFrame()
df["y"] = labels
df["Emb1"] = embeddings[:,0]
df["Emb2"] = embeddings[:,1]
#df.to_csv("saved_df.csv")

list_color = sns.color_palette(CityScapes_palette, 19)
palette = {trainid2name[i]:list_color[i] for i in range(19)}
sns.scatterplot(x="Emb1", y="Emb2", hue= [trainid2name[item] for item in df.y.tolist()],
                palette=palette,
                marker="s",  # Use square markers
                s=1,  # Size of markers
                data=df).set(title="Cityscapes features map")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.savefig("/home/s/kiendn/saved/DeepLabv2/07-21_22-58-cut_classmix_proto_feat_out/DeepLabv2_Feature_Visualization_rare_classes1.png", bbox_inches="tight", dpi=300)
plt.close()
"""