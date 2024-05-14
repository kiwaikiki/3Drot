import os

import torch
import cv2
import numpy as np
import kornia

from my_loss import angles2Rotation_Matrix, GS_transform, quaternion2Rotation_Matrix
from my_dataset import Dataset
from my_network import parse_command_line, load_model, Network_GS
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from shutil import copyfile


def infer(args):
    model = load_model(args)
    model.load_state_dict(torch.load(args.path_checkpoint, map_location='cuda:0'))
    model.eval()

    test_dataset = Dataset(args.path_pics, args.path_csv, 'test', args.input_width, args.input_height)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)
    print(len(test_loader))

    np.set_printoptions(suppress=True)
    done=0
    with torch.no_grad():
        with open(args.path_infer, 'w') as f:
            for sample in test_loader:
                preds = model(sample['pic'].cuda())
                gt_transforms = sample['transform']
    
                for i in range(args.batch_size):
                    index = sample['index'][i].item()
                    print(index)
                    print(20  * '*')
                    print("GT:")
                    gt_transform = gt_transforms[i].cpu().numpy()
                    print("Det: ", np.linalg.det(gt_transform))
                    print(gt_transform)
                    
                    transform = GS_transform((preds[0][i],preds[1][i])).cpu().numpy()[0]
                    # transform = angles2Rotation_Matrix(preds[i]).cpu().numpy()
                    # transform = quaternion2Rotation_Matrix(preds[i]).cpu().numpy()

                    print("Predict:")
                    print("Det: ", np.linalg.det(transform))
                    print(transform)
                    print(f'{index},{','.join(map(str, transform.ravel()))}', file=f)

if __name__ == '__main__':
    """
    Runs inference and writes prediction csv files.
    Example usage: python infer.py --no_preload -r 200 -iw 258 -ih 193 -b 32 /path/to/MLBinsDataset/EXR/dataset.json
    """
    # import os
    pic_dir = 'cool_cube'
    csv_dir = 'matice.csv'
    args = parse_command_line()
    args.path_pics = pic_dir
    args.path_csv = csv_dir
    args.input_width = 256
    args.input_height = 256
    args.batch_size = 4
    args.workers = 4
    args.repr = 'GS'
    args.path = args.repr + '/angle/'
    for i in range(100, 101, 10):
        args.path_checkpoint = f'siet/training_data/{pic_dir}/checkpoints/{args.path}checkpoint_{i:03d}.pth'
        args.path_infer = f'siet/training_data/{pic_dir}/inferences/{args.path}infer_results{i:03d}.csv'
        print(args.path_checkpoint)
        if not os.path.exists(args.path_checkpoint):
            print(f'Path {args.path_checkpoint} does not exist')
            break

        infer(args)
        print(f'Inference {i} done {args.path_checkpoint} {args.path_infer}')