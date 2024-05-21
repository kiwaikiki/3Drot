import torch
import numpy as np
import os
import resource 
import psutil
import gc


from network import normalized_l2_loss,  load_model, parse_command_line
from dataset import Dataset
from loss import (GS_Loss_Calculator, 
                     Axis_angle_4D_Loss_Calculator,
                     Stereographic_Loss_Calculator,
                     )
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)
gc.enable()

# select device
# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["NVIDIA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
print('Soft limit starts as  :', soft)
print('Hard limit starts as  :', hard)

limit = 1024**3 * 40
if hard != -1:
    limit = min(limit, hard)
resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))

soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
print('Soft limit starts as  :', soft)
print('Hard limit starts as  :', hard)

# exit()


def train(args):
    model = load_model(args)

    train_dataset = Dataset(args.json_path, 'train', args.input_width, args.input_height)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_dataset = Dataset(args.json_path, 'val', args.input_width, args.input_height)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_calculator = args.loss_calculator(args)
    if args.resume is not None:
        loss_calculator.load_results(args.resume)

    start_epoch = 0 if args.resume is None else args.resume

    print("Starting at epoch {}".format(start_epoch))
    print("Running till epoch {}".format(args.epochs))

  
    for e in range(start_epoch, args.epochs):
        # with open('train_err.out', 'a') as f:
        #     print("Starting epoch: ", e, file=f)
        print("Starting epoch: ", e)

        for sample in train_loader:
            with torch.autograd.set_detect_anomaly(True):
                # print('memory:    ', torch.cuda.memory_allocated())
                # print('cpu memory:', psutil.Process(os.getpid()).memory_info().rss)
                preds = model(sample['xyz'].cuda())

                optimizer.zero_grad()
                
                loss = loss_calculator.calculate_train_loss(preds, sample['bin_transform'][:, :3, :3].cuda())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_calculator.append_to_train_loss_all()
        with torch.no_grad():
            loss_calculator.reset_val_loss()

            for sample in val_loader:
                preds = model(sample['xyz'].cuda())
                optimizer.zero_grad()
                loss_calculator.calculate_val_loss(preds, sample['bin_transform'][:, :3, :3].cuda())

            loss_calculator.print_results(e, args.epochs)
            loss_calculator.append_to_val_loss_all()
                

        if args.dump_every != 0 and (e) % args.dump_every == 0:
            print("Saving checkpoint",  os.path.join(args.path_checkpoints, f'{e:03d}.pth'))
            torch.save(model.state_dict(), os.path.join(args.path_checkpoints, f'{e:03d}.pth'))
            loss_calculator.save_results(e)
        
        torch.cuda.empty_cache()
          
    loss_calculator.save_results()
    print("Finished training")



if __name__ == '__main__':
    """
    Example usage: python train.py -iw 1032 -ih 772 -b 12 -e 500 -de 10 -lr 1e-3 -bb resnet34 -w 0.1 /path/to/MLBinsDataset/EXR/dataset.json
    """
    args = parse_command_line()
    args.json_path = 'bins/VISIGRAPP_TRAIN/dataset.json'
    args.input_width = 258
    args.input_height = 193
    args.batch_size = 32
    args.workers = 4
    args.dump_every = 10
    args.epochs = 11

    loss_calcs = {
        'GS': GS_Loss_Calculator,
        'Axis_Angle_4D': Axis_angle_4D_Loss_Calculator,
        'Stereographic': Stereographic_Loss_Calculator
    }


    # args.dataset = 'cube_cool'
    # args.path_paxisics = f'datasets/{args.dataset}'
    args.repr = 'Axis_Angle_4D'
    args.loss_type = 'elements'
    args.loss_calculator = loss_calcs[args.repr]
    args.path_checkpoints = os.path.join('siet_for_bins', 'training_data', 'checkpoints', args.repr, args.loss_type)
    # # args.resume = 80
    train(args)

    reprs = [
        'GS',
        'Axis_Angle_4D',
        'Stereographic',
        # 'Matrix'
    ]
    losses = [
            'angle_rotmat',
            'elements',
            'angle_vectors'
              ]


