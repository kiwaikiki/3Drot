import torch
import numpy as np
import os
import resource 
import psutil
import gc


from my_network import normalized_l2_loss,  load_model, parse_command_line
from my_dataset import Dataset
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

def get_angles(pred, gt, sym_inv=False, eps=1e-7):
    """
    Calculates angle between pred and gt vectors.
    Clamping args in acos due to: https://github.com/pytorch/pytorch/issues/8069

    :param pred: tensor with shape (batch_size, 3)
    :param gt: tensor with shape (batch_size, 3)
    :param sym_inv: if True the angle is calculated w.r.t bin symmetry
    :param eps: float for NaN avoidance if pred is 0
    :return: tensor with shape (batch_size) containing angles
    """
    pred_norm = torch.norm(pred, dim=-1)
    gt_norm = torch.norm(gt, dim=-1)
    dot = torch.sum(pred * gt, dim=-1)
    if sym_inv:
        angles = torch.acos(torch.clamp(torch.abs(dot / (eps + pred_norm * gt_norm)), -1 + eps, 1 - eps))
    else:
        angles = torch.acos(torch.clamp(dot/(eps + pred_norm * gt_norm), -1 + eps, 1 - eps))
    return angles

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

    loss_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
    loss_z_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
    loss_y_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()

    l1_loss = torch.nn.L1Loss()

    start_epoch = 0 if args.resume is None else args.resume
    with open('train_err.out', 'a') as f:
        print("Starting at epoch {}".format(start_epoch), file=f)
        print("Running till epoch {}".format(args.epochs), file=f)
    print("Starting at epoch {}".format(start_epoch))
    print("Running till epoch {}".format(args.epochs))

    train_loss_all = []
    val_loss_all = []
    for e in range(start_epoch, args.epochs):
        with open('train_err.out', 'a') as f:
            print("Starting epoch: ", e, file=f)
        print("Starting epoch: ", e)

        for sample in train_loader:
            print('memory:    ', torch.cuda.memory_allocated())
            print('cpu memory:', psutil.Process(os.getpid()).memory_info().rss)
            pred_z, pred_y = model(sample['pic'].cuda())
            optimizer.zero_grad()
            
            # display_picture(sample['pic'][0].cpu().detach().numpy())

            # Angle loss is used for rotational components.
            loss_z = torch.mean(get_angles(pred_z, sample['bin_transform'][:, :3, 2].cuda()))
            # loss_y = torch.mean(get_angles(pred_y, sample['bin_transform'][:, :3, 1]#.cuda(), sym_inv=True))
            loss_y = torch.mean(get_angles(pred_y, sample['bin_transform'][:, :3, 1].cuda()))
            loss = loss_z + loss_y  

            # Note running loss calc makes loss increase in the beginning of training!
            loss_z_running = 0.9 * loss_z_running + 0.1 * loss_z
            loss_y_running = 0.9 * loss_y_running + 0.1 * loss_y
            loss_running = 0.9 * loss_running + 0.1 * loss.detach().item()
            with open('train_err.out', 'a') as f:
                print("Running loss: {}, z loss: {}, y loss: {}"
                      .format(loss_running.item(),  loss_z_running.item(), loss_y_running.item()), file=f)
            print("Running loss: {}, z loss: {}, y loss: {}"
                  .format(loss_running.item(),  loss_z_running.item(), loss_y_running.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_all.append(loss_running)
        with torch.no_grad():
            val_losses = []
            val_losses_z = []
            val_losses_y = []
            # val_angles = []
            # val_magnitudes = []

            for sample in val_loader:
                pred_z, pred_y = model(sample['pic'].cuda())
                optimizer.zero_grad()

                loss_z = torch.mean(get_angles(pred_z, sample['bin_transform'][:, :3, 2].cuda()))
                loss_y = torch.mean(get_angles(pred_y, sample['bin_transform'][:, :3, 1].cuda(), sym_inv=True))
                # loss_t = normalized_l2_loss(pred_t, sample['bin_translation'].cuda())
                loss = loss_z + loss_y 

                val_losses.append(loss.item())
                val_losses_z.append(loss_z.item())
                val_losses_y.append(loss_y.item())


            print(20 * "*")
            print("Epoch {}/{}".format(e, args.epochs))
            print("means - \t val loss: {} \t z loss: {} \t y loss: {}"
                  .format(np.mean(val_losses), np.mean(val_losses_z), np.mean(val_losses_y)))
            print("medians - \t val loss: {} \t z loss: {} \t y loss: {} "
                  .format(np.median(val_losses), np.median(val_losses_z), np.median(val_losses_y)))
            
            with open('train_err.out', 'a') as f:
                print(20 * "*", file=f)
                print("Epoch {}/{}".format(e, args.epochs), file=f)
                print("means - \t val loss: {} \t z loss: {} \t y loss: {}"
                      .format(np.mean(val_losses), np.mean(val_losses_z), np.mean(val_losses_y)), file=f)
                print("medians - \t val loss: {} \t z loss: {} \t y loss: {} "
                      .format(np.median(val_losses), np.median(val_losses_z), np.median(val_losses_y)), file=f)
           
            with open('train_err_just_ends.out', 'a') as f:
                print(20 * "*", file=f)
                print("Epoch {}/{}".format(e, args.epochs), file=f)
                print("means - \t val loss: {} \t z loss: {} \t y loss: {}"
                      .format(np.mean(val_losses), np.mean(val_losses_z), np.mean(val_losses_y)), file=f)
                print("medians - \t val loss: {} \t z loss: {} \t y loss: {} "
                      .format(np.median(val_losses), np.median(val_losses_z), np.median(val_losses_y)), file=f)
                
            val_loss_all.append(np.mean(val_losses))

            # del val_losses
            # del val_losses_z
            # del val_losses_y


        if args.dump_every != 0 and (e) % args.dump_every == 0:
            with open('train_err.out', 'a') as f:
                print("Saving checkpoint", file=f)
            print("Saving checkpoint")
            if not os.path.isdir('checkpoints/'):
                os.mkdir('checkpoints/')
            torch.save(model.state_dict(), 'checkpoints/{:03d}.pth'.format(e))
        
        torch.cuda.empty_cache()
        # clean also cpu memory
        
        



    np.set_printoptions(suppress=True)
    np.savetxt('train_err.out', train_loss_all.detach().numpy(), delimiter=',')
    np.savetxt('val_err.out', val_loss_all.detach().numpy(), delimiter=',')


if __name__ == '__main__':
    """
    Example usage: python train.py -iw 1032 -ih 772 -b 12 -e 500 -de 10 -lr 1e-3 -bb resnet34 -w 0.1 /path/to/MLBinsDataset/EXR/dataset.json
    """
    # args = parse_command_line()
    pic_dir = 'output/'
    csv_dir = 'matice.csv'
    args = parse_command_line()
    args.path_pics = pic_dir
    args.path_csv = csv_dir
    args.input_width = 256
    args.input_height = 256
    args.batch_size = 8
    args.workers = 4
    args.epochs = 100

    train(args)