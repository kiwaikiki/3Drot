import torch
import numpy as np
import os
import resource 
import psutil
import gc


from my_network import normalized_l2_loss,  load_model, parse_command_line
from my_dataset import Dataset
from my_loss import GSLossCalculator
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

gc.enable()

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

    loss_calculator = GSLossCalculator()
    # l1_loss = torch.nn.L1Loss()

    start_epoch = 0 if args.resume is None else args.resume

    print("Starting at epoch {}".format(start_epoch))
    print("Running till epoch {}".format(args.epochs))

  
    for e in range(start_epoch, args.epochs):
        with open('train_err.out', 'a') as f:
            print("Starting epoch: ", e, file=f)
        print("Starting epoch: ", e)

        for sample in train_loader:
            with torch.autograd.set_detect_anomaly(True):
                # print('memory:    ', torch.cuda.memory_allocated())
                # print('cpu memory:', psutil.Process(os.getpid()).memory_info().rss)
                pred_z, pred_y = model(sample['pic'].cuda())
                optimizer.zero_grad()
                
                # display_picture(sample['pic'][0].cpu().detach().numpy())
                loss = loss_calculator.calculate_angle_loss(pred_z, pred_y, sample['transform'].cuda())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_calculator.append_to_train_loss_all()
        with torch.no_grad():
           
            for sample in val_loader:
                pred_z, pred_y = model(sample['pic'].cuda())
                optimizer.zero_grad()
                loss_calculator.calculate_val_loss(pred_z, pred_y, sample['transform'].cuda())

            loss_calculator.print_results(e, args.epochs)
            loss_calculator.append_to_val_loss_all()
                

        if args.dump_every != 0 and (e) % args.dump_every == 0:
            with open('train_err.out', 'a') as f:
                print("Saving checkpoint", file=f)
            print("Saving checkpoint")
            # if not os.path.isdir(args.path_checkpoints):
            #     os.mkdir(args.path_checkpoints)
            torch.save(model.state_dict(), args.path_checkpoints + '{:03d}.pth'.format(e))
        
        torch.cuda.empty_cache()
        # clean also cpu memory
          
    loss_calculator.save_results()
    torch.save(model.state_dict(), 'checkpoints/final.pth')
    # release memory
    del model
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset
    del optimizer
    del loss_calculator
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    """
    Example usage: python train.py -iw 1032 -ih 772 -b 12 -e 500 -de 10 -lr 1e-3 -bb resnet34 -w 0.1 /path/to/MLBinsDataset/EXR/dataset.json
    """
    # args = parse_command_line()
    pic_dir = '../blendre/output/'
    csv_dir = '../blendre/matice.csv'
    args = parse_command_line()
    args.path_pics = pic_dir
    args.path_csv = csv_dir
    args.input_width = 256
    args.input_height = 256
    args.batch_size = 32
    args.workers = 4
    args.dump_every = 10
    args.repr = 'GS'
    args.loss_type = 'angle'
    args.path_checkpoints = f'checkpoints/{args.repr}/{args.loss_type}/'

    args.epochs = 100

    train(args)