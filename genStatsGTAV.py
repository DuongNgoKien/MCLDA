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
    pil_label = Image.open(file)
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
    parser.add_argument('--gta_path', help='gta data path')
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
