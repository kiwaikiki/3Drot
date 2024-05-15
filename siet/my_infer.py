import os

import torch
import cv2
import numpy as np
import kornia

from my_loss import angles2Rotation_Matrix, quaternion2Rotation_Matrix, angle_bins2rotation_Matrix
from my_dataset import Dataset
from my_network import parse_command_line, load_model, Network_GS
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from shutil import copyfile

def GS_transform(preds):
    pred_z, pred_y = preds
    z = pred_z / torch.linalg.norm(pred_z)
    y = pred_y - torch.dot(z, pred_y)*z
    y = y / torch.linalg.norm(y)
    x = torch.linalg.cross(y, z)

    transform = torch.zeros([3, 3])
    transform[:, 0] = x
    transform[:, 1] = y
    transform[:, 2] = z
    return transform

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
                    
                    if args.repr == 'GS':
                        transform = args.repr_f((preds[0][i],preds[1][i])).cpu().numpy()
                    else:
                        transform = args.repr_f(preds[i]).cpu().numpy()

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
    args = parse_command_line()
    args.path_csv = 'matice.csv'
    args.input_width = 256
    args.input_height = 256
    args.batch_size = 4
    args.workers = 4

    repr_func = {
        'GS' : GS_transform,
        'Euler' : angles2Rotation_Matrix,
        'Quaternion' : quaternion2Rotation_Matrix,
        'Euler_binned' : angle_bins2rotation_Matrix
    }

    args.path_pics = 'cube_big_hole'
    args.repr = 'GS'
    args.loss_type = 'elements'
    args.path = os.path.join(args.repr, args.loss_type)
    args.repr_f = repr_func[args.repr]
    for i in range(0, 101, 10):
        args.path_checkpoint = f'siet/training_data/{args.path_pics}/checkpoints/{args.path}/{i:03d}.pth'
        args.path_infer = f'siet/training_data/{args.path_pics}/inferences/{args.path}/infer_results{i:03d}.csv'
        print(args.path_checkpoint)
        if not os.path.exists(args.path_checkpoint):
            print(f'Path {args.path_checkpoint} does not exist')
            break

        infer(args)
        print(f'Inference {i} done {args.path_checkpoint} {args.path_infer}')