import os

import torch
import cv2
import numpy as np
from my_dataset import Dataset

from my_network import Network, parse_command_line, load_model
from scipy.spatial.transform.rotation import Rotation
from torch.utils.data import DataLoader
from shutil import copyfile


def infer(args):
    model = load_model(args).eval()

    test_dataset = Dataset(args.path_pics, args.path_csv, 'test', args.input_width, args.input_height)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)

    np.set_printoptions(suppress=True)

    with torch.no_grad():
        with open('infer_results.csv', 'w') as f:
            for sample in test_loader:
                pred_zs, pred_ys = model(sample['pic'].cuda())
                gt_transforms = sample['transform']

                for i in range(len(pred_zs)):
                    index = sample['index'][i].item()
                    print(index)
                    print(20  * '*')
                    print("GT:")
                    gt_transform = gt_transforms[i].cpu().numpy()
                    print("Det: ", np.linalg.det(gt_transform))
                    print(gt_transform)

                    z = pred_zs[i].cpu().numpy()
                    z /= np.linalg.norm(z)

                    y = pred_ys[i].cpu().numpy()
                    y = y - np.dot(z, y)*z
                    y /= np.linalg.norm(y)

                    x = np.cross(y, z)

                    transform = np.zeros([3, 3])
                    transform[:3, 0] = x
                    transform[:3, 1] = y
                    transform[:3, 2] = z

                    print("Predict:")
                    print("Det: ", np.linalg.det(transform))
                    print(transform)
                    print(f'{index},{','.join(map(str, transform.ravel()))}', file=f)

if __name__ == '__main__':
    """
    Runs inference and writes prediction csv files.
    Example usage: python infer.py --no_preload -r 200 -iw 258 -ih 193 -b 32 /path/to/MLBinsDataset/EXR/dataset.json
    """
    pic_dir = '../blendre/output/'
    csv_dir = '../blendre/matice.csv'
    args = parse_command_line()
    args.path_pics = pic_dir
    args.path_csv = csv_dir
    args.input_width = 256
    args.input_height = 256
    args.batch_size = 32
    args.workers = 4

    infer(args)
