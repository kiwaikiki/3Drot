import torch
import numpy as np

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

# kedy get_angles kedy angle

def angle(x, y):
    '''
    Calculate the angle between two vectors x and y
    '''
    x = x.ravel()
    y = y.ravel()
    return torch.rad2deg(torch.arccos(torch.dot(x, y) / (torch.norm(x) * torch.norm(y))))

def rotation_angle(R):
    '''
    Calculate the rotation angle of a single tranformation by a 3x3 rotation matrix R
    '''
    return torch.rad2deg(torch.arccos(torch.clip((torch.trace(R) - 1) / 2, -1, 1)))

def calculate_eTE(R_gt, R_est):
    '''
    Calculate the angle between two rotation matrices R_gt and R_est
    ''' 
    # angles = np..apply_along_axis(lambda Rt, Re: rotation_angle(Rt.T @ Re), 1, R_gt, R_est)
    angles = torch.stack([rotation_angle(R_gt[i].T @ R_est[i]) for i in range(len(R_gt))])
    return torch.mean(angles).cuda()

def angles2Rotation_Matrix(angles):
    '''
    Convert angles to a rotation matrix
    '''
    x, y, z = angles
    Rx = torch.tensor([[1, 0, 0], [0, torch.cos(x), -torch.sin(x)], [0, torch.sin(x), torch.cos(x)]], requires_grad=True)
    Ry = torch.tensor([[torch.cos(y), 0, torch.sin(y)], [0, 1, 0], [-torch.sin(y), 0, torch.cos(y)]], requires_grad=True)
    Rz = torch.tensor([[torch.cos(z), -torch.sin(z), 0], [torch.sin(z), torch.cos(z), 0], [0, 0, 1]], requires_grad=True)
    return torch.matmul(Rz, torch.matmul(Ry, Rx)).cuda()

def angles2Rotation_Matrix_for_batches(angles):
    '''
    Convert batch of angles to rotation matrices
    '''
    # angles is a tensor of shape (batch_size, 3)
    x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]
    matrices = torch.stack([angles2Rotation_Matrix([x[i], y[i], z[i]]) for i in range(len(x))])
    return matrices

def rotation_Matrix2angles(R):
    '''
    Convert a rotation matrix to angles
    '''
    x = torch.arctan2(R[2, 1], R[2, 2])
    y = torch.arctan2(-R[2, 0], torch.sqrt(R[2, 1]**2 + R[2, 2]**2))
    z = torch.arctan2(R[1, 0], R[0, 0])
    return torch.tensor([x, y, z], requires_grad=True).cuda()

def GS_transform(preds):
    '''
    Calculate the transformation matrix from the predicted vectors
    '''
    pred_z, pred_y = preds
    z = pred_z
    z = z / torch.linalg.norm(z)
    
    y = pred_y
    # y = y - torch.dot(z, y)*z
    y = y - torch.sum(z * y, dim=-1, keepdim=True) * z
    y = y / torch.linalg.norm(y)

    x = torch.linalg.cross(y, z)

    transform = torch.zeros([pred_y.shape[0], 3, 3])
    transform[:, :3, 0] = x
    transform[:, :3, 1] = y
    transform[:, :3, 2] = z

    return transform


class LossCalculator:
    def __init__(self, loss_type='angle'):
        self.loss_type = loss_type

        self.loss_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
        
        self.train_loss_all = []
        self.val_loss_all = []

        self.val_losses = []

    def reset_val_loss(self):
        self.val_losses = []

    def calculate_train_loss(self, preds, true_transform):
        return self.function_dict[self.loss_type][0](preds, true_transform)
    
    def calculate_val_loss(self, preds, true_transform):
        return self.function_dict[self.loss_type][1](preds, true_transform)

    def append_to_train_loss_all(self):
        self.train_loss_all.append(self.loss_running.detach().item())

    def append_to_val_loss_all(self):
        self.val_loss_all.append(np.mean(self.val_losses))
        

class GSLossCalculator(LossCalculator):
    def __init__(self, loss_type='angle'):
        super().__init__(loss_type)
        self.loss_z_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
        self.loss_y_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
        
        self.val_losses_z = []
        self.val_losses_y = []

        self.function_dict = {'angle': (self.calculate_angle_loss, 
                                    self.calculate_val_angle_loss),
                        'elements': (self.calculate_elements_train_loss,
                                    self.calculate_elements_val_loss)}

        self.folder = 'results/GS/' + loss_type + '/'
        self.train_file = self.folder + 'train_err.out'
        self.val_file = self.folder + 'val_err.out'
        # self.all_running = self.folder + 'train_err_running2.out'
        # self.just_ends_file = self.folder + 'train_err_just_ends2.out'

    def calculate_angle_loss(self, preds, true_transform):
        transform = GS_transform(preds)
        true_transfrm = true_transform.float()
        loss = torch.mean(calculate_eTE(true_transfrm, transform.cuda()))

        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss
    
    def calculate_val_angle_loss(self, preds, true_transform):
        self.val_losses.append(self.calculate_angle_loss(preds, true_transform).detach().item())

    def reset_val_loss(self):
        self.val_losses = []
        self.val_losses_z = []
        self.val_losses_y = []

    def calculate_elements_train_loss(self, preds, transform): 
        pred_z, pred_y = preds
        loss_z = torch.mean(get_angles(pred_z, transform[:, :3, 2].cuda()))
        loss_y = torch.mean(get_angles(pred_y, transform[:, :3, 1].cuda()))
        loss = loss_z + loss_y  

        # Note running loss calc makes loss increase in the beginning of training!
        self.loss_z_running = 0.9 * self.loss_z_running + 0.1 * loss_z
        self.loss_y_running = 0.9 * self.loss_y_running + 0.1 * loss_y
        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        self.print_running_loss()
        return loss
    
    def calculate_elements_val_loss(self, preds, transform): 
        pred_z, pred_y = preds
        loss_z = torch.mean(get_angles(pred_z, transform[:, :3, 2].cuda()))
        loss_y = torch.mean(get_angles(pred_y, transform[:, :3, 1].cuda(), sym_inv=True))
        loss = loss_z + loss_y 

        self.val_losses.append(loss.detach().item())
        self.val_losses_z.append(loss_z.detach().item())
        self.val_losses_y.append(loss_y.detach().item())

   
    def print_running_loss(self):
        # with open(self.all_running, 'a') as f:
        #     print("Running loss: {}, z loss: {}, y loss: {}"
        #         .format(self.loss_running.item(),  self.loss_z_running.item(), self.loss_y_running.item()), file=f)
        print("Running loss: {}, z loss: {}, y loss: {}"
            .format(self.loss_running.item(),  self.loss_z_running.item(), self.loss_y_running.item()))
    
    def print_results(self, e, epochs):
        print(20 * "*")
        print("Epoch {}/{}".format(e, epochs))
        print("means - \t val loss: {} \t z loss: {} \t y loss: {}"
                .format(np.mean(self.val_losses), np.mean(self.val_losses_z), np.mean(self.val_losses_y)))
        print("medians - \t val loss: {} \t z loss: {} \t y loss: {} "
                .format(np.median(self.val_losses), np.median(self.val_losses_z), np.median(self.val_losses_y)))
    
        # with open(self.just_ends_file, 'a') as f:
        #     print(20 * "*", file=f)
        #     print("Epoch {}/{}".format(e, epochs), file=f)
        #     print("means - \t val loss: {} \t z loss: {} \t y loss: {}"
        #         .format(np.mean(self.val_losses), np.mean(self.val_losses_z), np.mean(self.val_losses_y)), file=f)
        #     print("medians - \t val loss: {} \t z loss: {} \t y loss: {} "
        #         .format(np.median(self.val_losses), np.median(self.val_losses_z), np.median(self.val_losses_y)), file=f)
    
    def save_results(self):
        np.set_printoptions(suppress=True)
        np.savetxt(self.train_file, self.train_loss_all, delimiter=',')
        np.savetxt(self.val_file, self.val_loss_all, delimiter=',')


class EulerLossCalculator(LossCalculator):
    def __init__(self, loss_type='angle'):
        super().__init__(loss_type)

        self.function_dict = {'angle': (self.calculate_angle_loss, 
                                    self.calculate_val_angle_loss),
                        'elements': (self.calculate_elements_loss,
                                    self.calculate_val_elements_loss)}

        self.folder = f'results/Euler/{loss_type}/'
        self.train_file = self.folder + 'train_err2.out'
        self.val_file = self.folder + 'val_err2.out'


    def calculate_angle_loss(self, pred, gt):
        pred_R = angles2Rotation_Matrix_for_batches(pred)
        true_transfrm = gt.float()

        loss = torch.mean(calculate_eTE(true_transfrm, pred_R.cuda())).cuda()
        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss
    
    def calculate_val_angle_loss(self, pred, gt):
        self.val_losses.append(self.calculate_angle_loss(pred, gt).detach().item())
           
    
    def calculate_elements_loss(self, preds, true_transform):
        gt_angles = torch.stack([rotation_Matrix2angles(true_transform[i]) for i in range(len(true_transform))])

        loss = torch.mean(torch.abs(preds - gt_angles)).cuda()

        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss
    
    def calculate_val_elements_loss(self, preds, true_transform):
        loss = self.calculate_elements_loss(preds, true_transform)
        self.val_losses.append(loss.detach().item())
        
    
    def print_results(self, e, epochs):
        print(20 * "*")
        print("Epoch {}/{}".format(e, epochs))
        print("mean - val loss: {}".format(np.mean(self.val_losses)))
        print("median - val loss: {}".format(np.median(self.val_losses)))

    def save_results(self):
        np.set_printoptions(suppress=True)
        np.savetxt(self.train_file, self.train_loss_all, delimiter=',')
        np.savetxt(self.val_file, self.val_loss_all, delimiter=',')


class QuaternionLossCalculator(LossCalculator):
    def __init__(self, loss_type='angle'):
        super().__init__(loss_type)

        self.function_dict = {'angle': (self.calculate_angle_loss, 
                                    self.calculate_val_angle_loss),
                        'elements': (self.calculate_elements_loss,
                                    self.calculate_val_elements_loss)}

        self.folder = f'results/Quaternion/{loss_type}/'
        self.train_file = self.folder + 'train_err.out'
        self.val_file = self.folder + 'val_err.out'