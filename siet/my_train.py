import torch
import numpy as np
import os
import resource 
import psutil
import gc


from my_network import normalized_l2_loss,  load_model, parse_command_line
from my_dataset import Dataset
from my_loss import GS_Loss_Calculator, Euler_Loss_Calculator, Euler_binned_Loss_Calculator, Quaternion_Loss_Calculator, Axis_angle_Loss_Calculator
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)
gc.enable()

# select device
# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

def display_picture(pic):
    pic = np.transpose(pic, [1, 2, 0])
    plt.imshow(pic)
    plt.show()

def train(args):
    model = load_model(args)

    train_dataset = Dataset(args.path_pics, args.path_csv, 'train', args.input_width, args.input_height)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_dataset = Dataset(args.path_pics, args.path_csv, 'val', args.input_width, args.input_height)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_calculator = args.loss_calculator(args.loss_type)

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
                preds = model(sample['pic'].cuda())

                optimizer.zero_grad()
                
                # display_picture(sample['pic'][0].cpu().detach().numpy())
                loss = loss_calculator.calculate_train_loss(preds, sample['transform'].cuda())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_calculator.append_to_train_loss_all()
        with torch.no_grad():
            loss_calculator.reset_val_loss()

            for sample in val_loader:
                preds = model(sample['pic'].cuda())
                optimizer.zero_grad()
                loss_calculator.calculate_val_loss(preds, sample['transform'].cuda())

            loss_calculator.print_results(e, args.epochs)
            loss_calculator.append_to_val_loss_all()
                

        if args.dump_every != 0 and (e) % args.dump_every == 0:
            # with open('train_err.out', 'a') as f:
            #     print("Saving checkpoint", file=f)
            print("Saving checkpoint",  args.path_checkpoints + '{:03d}.pth'.format(e))
            # if not os.path.isdir(args.path_checkpoints):
            #     os.mkdir(args.path_checkpoints)
            torch.save(model.state_dict(), args.path_checkpoints + '{:03d}.pth'.format(e))
        
        torch.cuda.empty_cache()
        # clean also cpu memory
          
    loss_calculator.save_results()
    # release memory



if __name__ == '__main__':
    """
    Example usage: python train.py -iw 1032 -ih 772 -b 12 -e 500 -de 10 -lr 1e-3 -bb resnet34 -w 0.1 /path/to/MLBinsDataset/EXR/dataset.json
    """
    # args = parse_command_line()p
    pic_dir = '../blendre/output/'
    csv_dir = '../blendre/matice.csv'
    args = parse_command_line()
    args.path_pics = pic_dir
    args.path_csv = csv_dir
    args.input_width = 256
    args.input_height = 256
    args.batch_size = 64
    args.workers = 4
    args.dump_every = 10
    args.repr = 'Euler'
    args.loss_type = 'angle'
    loss_calcs = {
        'GS': GS_Loss_Calculator,
        'Euler': Euler_Loss_Calculator,
        'Euler_binned': Euler_binned_Loss_Calculator,
        'Quaternion': Quaternion_Loss_Calculator,
        'Axis-angle': Axis_angle_Loss_Calculator
    }
    args.loss_calculator = loss_calcs[args.repr]
    args.path_checkpoints = f'checkpoints/{args.repr}/{args.loss_type}/'
    # args.resume = 70
    args.epochs = 101

    train(args)