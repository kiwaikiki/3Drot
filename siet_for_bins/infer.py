import os

import torch
import cv2
import numpy as np
import kornia

from dataset import Dataset
from network import parse_command_line, load_model
from scipy.spatial.transform.rotation import Rotation
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

def stereo2rotation_Matrix(preds):
    first, last = preds[:2], preds[2:]
    norm_last = torch.linalg.norm(last)
    new_element = (norm_last**2 - 1) /2 
    print(new_element)
    new_last = torch.tensor([new_element, last[0], last[1], last[2]]).cuda()
    new_last = new_last / norm_last
    gs = torch.cat([first, new_last])
    transform = GS_transform({gs[:3], gs[3:]})
    return transform

def axis_angle3D2rotation_Matrix(axis):
    return kornia.geometry.conversions.axis_angle_to_rotation_matrix(torch.stack([axis]))[0]

def axis_angle4D2rotation_Matrix(pred):
    axis, angle = pred
    axis = axis / torch.linalg.norm(axis)
    return kornia.geometry.conversions.axis_angle_to_rotation_matrix(torch.stack([axis*angle]))[0]


def infer(args, export_to_folder=False):
    model = load_model(args).eval()
    model.load_state_dict(torch.load(args.path_checkpoint, map_location='cuda:0'))
    model.eval()

    dir_path = os.path.dirname(args.path)

    test_dataset = Dataset(args.json_path, 'test', args.input_width, args.input_height, preload=not args.no_preload)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    np.set_printoptions(suppress=True)

    with torch.no_grad():
        with open(args.path_infer, 'w') as f:
            for sample in test_loader:
                preds = model(sample['xyz'].cuda())
                gt_transforms = sample['bin_transform']
                filenames = sample['txt_path']
                for i in range(len(gt_transforms)):
                    print(20  * '*')
                    print("GT:")
                    gt_transform = gt_transforms[i].cpu().numpy()[:3, :3]
                    print("Det: ", np.linalg.det(gt_transform))
                    print(gt_transform)

                    if args.repr in ['GS', 'Axis_Angle_4D', 'Axis_Angle_binned']:
                        transform = args.repr_f((preds[0][i],preds[1][i])).cpu().numpy()
                    else:
                        transform = args.repr_f(preds[i]).cpu().numpy()

                    print("Predict:")
                    print("Det: ", np.linalg.det(transform))
                    print(transform)
                    # print(filenames[i] + ',' + ','.join(map(str, transform.ravel())), file=f)
                    print(','.join(map(str, transform.ravel())), file=f)



if __name__ == '__main__':
    """
    Runs inference and writes prediction txt files.
    Example usage: python infer.py --no_preload -r 200 -iw 258 -ih 193 -b 32 /path/to/MLBinsDataset/EXR/dataset.json
    """
    args = parse_command_line()
    args.input_width = 258
    args.input_height = 193
    args.batch_size = 4
    args.workers = 4
    repr_func = {
        'GS' : GS_transform,
        'Axis_Angle_4D' : axis_angle4D2rotation_Matrix,
        'Axis_Angle_3D' : axis_angle3D2rotation_Matrix,
        'Stereographic' : stereo2rotation_Matrix,}
    
    args.json_path = 'bins/VISIGRAPP_TEST/dataset.json'   

    args.repr = 'Stereographic' # NOTE: change this to the representation you want to evaluate
    args.loss_type = 'angle_rotmat' # NOTE: change this to the loss function you want to use
    args.path = os.path.join(args.repr, args.loss_type)
    args.repr_f = repr_func[args.repr]

    args.path_checkpoint = f'siet_for_bins/training_data_synth/checkpoints/{args.path}/best.pth'
    args.path_infer = f'siet_for_bins/training_data_synth/inferences/{args.path}/infer_results_best.csv'
    if not os.path.exists(args.path_checkpoint):
        print(f'Path {args.path_checkpoint} does not exist')
       

    infer(args)
    print(f'Inference best done {args.path_checkpoint} {args.path_infer}')



    # for i in range(100, 201, 10):
    #     args.path_checkpoint = f'siet_for_bins/training_data_synth/checkpoints/{args.path}/{i:03d}.pth'
    #     args.path_infer = f'siet_for_bins/training_data_synth/inferences/{args.path}/infer_results{i:03d}.csv'
    #     if not os.path.exists(args.path_checkpoint):
    #         print(f'Path {args.path_checkpoint} does not exist')
    #         break

    #     infer(args)
    #     print(f'Inference {i} done {args.path_checkpoint} {args.path_infer}')


