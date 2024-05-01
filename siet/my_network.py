import argparse
import os

import torch
import torchvision

def normalized_l2_loss(pred, gt, reduce=True):
    """
    Returns normalized L2 loss: ||pred - gt||^2 / ||gt||^2
    :param pred: prediction vectors with shape (batch, n)
    :param gt: gt vectors with shape (batch, n)
    :param reduce: if True returns means of losses otherwise returns loss for each element of batch
    :return: normalized L2 loss
    """
    norm = torch.sum(gt ** 2, dim=-1) + 1e-7
    loss = torch.sum((pred - gt) ** 2, dim=-1) / norm
    if reduce:
        return torch.mean(loss)
    else:
        return loss



class Network_GS(torch.nn.Module):
    def __init__(self, backbone='resnet18'):
        super(Network_GS, self).__init__()

        
        pretrained_backbone_model = torchvision.models.resnet18(pretrained=True)
    
        last_feat = list(pretrained_backbone_model.children())[-1].in_features // 2

        self.backbone = torch.nn.Sequential(*list(pretrained_backbone_model.children())[:-3])

        
        self.fc_z = torch.nn.Sequential(torch.nn.Linear(last_feat, 128),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Linear(128, 64),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Linear(64, 3))

        self.fc_y = torch.nn.Sequential(torch.nn.Linear(last_feat, 128),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Linear(128, 64),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Linear(64, 3))

        # self.fc_t = torch.nn.Sequential(torch.nn.Linear(last_feat, 128),
        #                                 torch.nn.LeakyReLU(),
        #                                 torch.nn.Linear(128, 64),
        #                                 torch.nn.LeakyReLU(),
        #                                 torch.nn.Linear(64, 3))

    def forward(self, x):
        # x = self.init_conv(x)
        x = self.backbone(x)

        # Global Avg Pool
        x = torch.mean(x, -1)
        x = torch.mean(x, -1)

        # Max pooling
        # x = torch.max(x, -1)[0]
        # x = torch.max(x, -1)[0]

        z = self.fc_z(x)
        y = self.fc_y(x)

        return z, y

class Network_Euler(torch.nn.Module):
    def __init__(self, backbone='resnet18'):
        super(Network_Euler, self).__init__()

        
        pretrained_backbone_model = torchvision.models.resnet18(pretrained=True)
    
        last_feat = list(pretrained_backbone_model.children())[-1].in_features // 2

        self.backbone = torch.nn.Sequential(*list(pretrained_backbone_model.children())[:-3])

        
        self.angles = torch.nn.Sequential(torch.nn.Linear(last_feat, 128),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Linear(128, 64),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Linear(64, 3))

    def forward(self, x):
        # x = self.init_conv(x)
        x = self.backbone(x)

        # Global Avg Pool
        x = torch.mean(x, -1)
        x = torch.mean(x, -1)

        # Max pooling
        # x = torch.max(x, -1)[0]
        # x = torch.max(x, -1)[0]

        angles = self.angles(x)

        return angles


def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-r', '--resume', type=int, default=None, help='checkpoint to resume from')
    parser.add_argument('-nw', '--workers', type=int, default=0, help='workers')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--no_preload', action='store_true', default=False)
    parser.add_argument('-iw', '--input_width', type=int, default=256, help='size of input')
    parser.add_argument('-ih', '--input_height', type=int, default=256, help='size of input')
    parser.add_argument('-e', '--epochs', type=int, default=250, help='max number of epochs')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('-bb', '--backbone', type=str, default='resnet18', help='which backbone to use: resnet18/34/50')
    parser.add_argument('-de', '--dump_every', type=int, default=0, help='save every n frames during extraction scripts')
    parser.add_argument('-w', '--weight', type=float, default=1.0, help='weight for translation component')
    parser.add_argument('-ns', '--noise_sigma', type=float, default=None)
    parser.add_argument('-ts', '--t_sigma', type=float, default=0.0)
    parser.add_argument('-rr', '--random_rot', action='store_true', default=False)
    # parser.add_argument('path')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args


def load_model(args):
    """
    Loads model. If args.resum is None weights for the backbone are pre-trained on ImageNet, otherwise previous
    checkpoint is loaded
    """
    which_repr = {
        'GS': Network_GS,
        'Euler': Network_Euler
    }
    repr_network = which_repr[args.repr] 
    model = repr_network(backbone=args.backbone).cuda()
    # model = Network_Euler(backbone=args.backbone).cuda()
    # if args.resume is not None:
    #     sd_path = 'checkpoints/{:03d}.pth'.format(args.resume)
    #     print("Resuming from: ", sd_path)
    #     model.load_state_dict(torch.load(sd_path))
    return model
