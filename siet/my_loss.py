import torch
import numpy as np

class LossCalculator:
    def __init__(self, method: str):
        self.methods = {'grammschmidt': (self.GS_init, self.calculate_GS_train_loss)}
        init, calc = self.methods[method]
        init()
        self.calc = calc

        
    def GS_init(self):
        self.loss_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
        self.loss_z_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
        self.loss_y_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
        
        self.train_loss_all = []
        self.val_loss_all = []

        self.val_losses = []
        self.val_losses_z = []
        self.val_losses_y = []

        self.folder = 'results/'
        self.just_ends_file = self.folder + 'GS_train_err_just_ends.out'
        self.train_file = self.folder + 'GS_train_err.out'
        self.val_file = self.folder + 'GS_val_err.out'
        self.all_running = self.folder + 'GS_train_err_running.out'


    def calculate_GS_train_loss(self, pred_z, pred_y, transform): 
        loss_z = torch.mean(get_angles(pred_z,transform[:, :3, 2].cuda()))
        loss_y = torch.mean(get_angles(pred_y, transform[:, :3, 1].cuda()))
        loss = loss_z + loss_y  

        # Note running loss calc makes loss increase in the beginning of training!
        self.loss_z_running = 0.9 * self.loss_z_running + 0.1 * loss_z
        self.loss_y_running = 0.9 * self.loss_y_running + 0.1 * loss_y
        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        self.print_running_loss()
        return loss
    
    def append_to_train_loss_all(self):
        self.train_loss_all.append(self.loss_running.detach().item())

    def calculate_GS_val_loss(self, pred_z, pred_y, transform): 
        loss_z = torch.mean(get_angles(pred_z, transform[:, :3, 2].cuda()))
        loss_y = torch.mean(get_angles(pred_y, transform[:, :3, 1].cuda(), sym_inv=True))
        # loss_t = normalized_l2_loss(pred_t, sample['bin_translation'].cuda())
        loss = loss_z + loss_y 

        self.val_losses.append(loss.detach().item())
        self.val_losses_z.append(loss_z.detach().item())
        self.val_losses_y.append(loss_y.detach().item())

    def append_to_val_loss_all(self):
        self.val_loss_all.append(np.mean(self.val_losses))
    
    def print_running_loss(self):
        with open(self.all_running, 'a') as f:
            print("Running loss: {}, z loss: {}, y loss: {}"
                .format(self.loss_running.item(),  self.loss_z_running.item(), self.loss_y_running.item()), file=f)
        print("Running loss: {}, z loss: {}, y loss: {}"
            .format(self.loss_running.item(),  self.loss_z_running.item(), self.loss_y_running.item()))
    
    def print_results(self, e, epochs):
        print(20 * "*")
        print("Epoch {}/{}".format(e, epochs))
        print("means - \t val loss: {} \t z loss: {} \t y loss: {}"
                .format(np.mean(self.val_losses), np.mean(self.val_losses_z), np.mean(self.val_losses_y)))
        print("medians - \t val loss: {} \t z loss: {} \t y loss: {} "
                .format(np.median(self.val_losses), np.median(self.val_losses_z), np.median(self.val_losses_y)))
    
        with open(self.just_ends_file, 'a') as f:
            print(20 * "*", file=f)
            print("Epoch {}/{}".format(e, epochs), file=f)
            print("means - \t val loss: {} \t z loss: {} \t y loss: {}"
                .format(np.mean(self.val_losses), np.mean(self.val_losses_z), np.mean(self.val_losses_y)), file=f)
            print("medians - \t val loss: {} \t z loss: {} \t y loss: {} "
                .format(np.median(self.val_losses), np.median(self.val_losses_z), np.median(self.val_losses_y)), file=f)
    
    def save_results(self):
        np.set_printoptions(suppress=True)
        np.savetxt(self.train_file, self.train_loss_all, delimiter=',')
        np.savetxt(self.val_file, self.val_loss_all, delimiter=',')

# toto plati pre GS?
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