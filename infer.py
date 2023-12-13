import argparse
import numpy as np
import os
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data

from model.segformer import SegFormer
from model.deeplabv2 import Res_Deeplab
from data import loader

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
colors = [  # [  0,   0,   0],
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
def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = colors[l][0]
        g[temp == l] = colors[l][1]
        b[temp == l] = colors[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UDA evaluation script")
    parser.add_argument("-m","--model-path", type=str, default=None, required=True,
                        help="Model to evaluate")
    parser.add_argument("--model_type", help="Inference on Cityscapes or Naver images")
    parser.add_argument("--img_dir", help="Image folder")
    parser.add_argument("--save_dir", help="Output folder")
    return parser.parse_args()


def predict(model, img_dir=None, save_dir=None):

    dataset = loader(img_dir, is_transform=True, img_mean=IMG_MEAN)
    testloader = data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    for _, batch in enumerate(testloader):
        image, image_name = batch
        image_name = image_name[0].split("/")[-1]
        with torch.no_grad():
            _, output  = model(Variable(image).cuda())
            interp = nn.Upsample(size=image.shape[2:], mode='bilinear', align_corners=True)
            output = interp(output)

            output = output.cpu().data[0].numpy()
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int32)
        if save_dir:
            np.save(os.path.join(save_dir, image_name[:-4] + ".npy"), output)
            color_map = decode_segmap(output.squeeze()).astype(np.uint8)
            color_map = Image.fromarray(color_map)
            color_map.save(os.path.join(save_dir, image_name[:-4] + "_color.png"))

def main():
    """Create the model and start the evaluation process."""

    #gpu0 = args.gpu

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.model_type == "cityscapes-DeepLabv2": 
        model = Res_Deeplab(num_classes=19)
    elif args.model_type == "cityscapes-b5":
        model = SegFormer("B5", num_classes=19)
    elif args.model_type == "cityscapes-b3":
        model = SegFormer("B3", num_classes=19)
    elif args.model_type == "naver":
        model = SegFormer("B5", num_classes=18)
    else:
        raise ValueError(f'Model {args.model_type} is not supported')

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'])

    model.cuda()
    model.eval()

    predict(model, args.img_dir, args.save_dir)


if __name__ == '__main__':
    args = get_arguments()
    main()
