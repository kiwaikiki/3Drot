import torch
import numpy as np
import kornia

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

def calculate_eRE(R_gt, R_est):
    '''
    Calculate the angle between two rotation matrices R_gt and R_est
    ''' 
    # angles = np..apply_along_axis(lambda Rt, Re: rotation_angle(Rt.T @ Re), 1, R_gt, R_est)
    angls = [rotation_angle(torch.matmul(R_gt[i].T, R_est[i])) for i in range(len(R_gt))]
    angles = torch.stack([rotation_angle(torch.matmul(R_gt[i].T, R_est[i])) for i in range(len(R_gt))])
    # print(angles)
    return torch.mean(angles)

def angles2Rotation_Matrix(angles):
    '''
    Convert angles to a rotation matrix
    # '''
    # x, y, z = angles
    # Rx = torch.tensor([[1, 0, 0], [0, torch.cos(x), -torch.sin(x)], [0, torch.sin(x), torch.cos(x)]], requires_grad=True)
    # Ry = torch.tensor([[torch.cos(y), 0, torch.sin(y)], [0, 1, 0], [-torch.sin(y), 0, torch.cos(y)]], requires_grad=True)
    # Rz = torch.tensor([[torch.cos(z), -torch.sin(z), 0], [torch.sin(z), torch.cos(z), 0], [0, 0, 1]], requires_grad=True)
    # return torch.matmul(Rz, torch.matmul(Ry, Rx)).cuda()
    q = kornia.geometry.conversions.quaternion_from_euler(angles)
    return kornia.geometry.conversions.quaternion_to_rotation_matrix(q)

def angles2Rotation_Matrix_for_batches(angles):
    '''
    Convert batch of angles to rotation matrices
    '''
    # angles is a tensor of shape (batch_size, 3)
    # x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]
    # matrices = torch.stack([angles2Rotation_Matrix([x[i], y[i], z[i]]) for i in range(len(x))])
    # return matrices

    q = torch.stack(kornia.geometry.conversions.quaternion_from_euler(angles.T[0], angles.T[1], angles.T[2])).T
    return kornia.geometry.conversions.quaternion_to_rotation_matrix(q)

def rotation_Matrix2angles(R):
    '''
    Convert a rotation matrix to angles
    '''
    # x = torch.arctan2(R[2, 1], R[2, 2])
    # y = torch.arctan2(-R[2, 0], torch.sqrt(R[2, 1]**2 + R[2, 2]**2))
    # z = torch.arctan2(R[1, 0], R[0, 0])
    # return torch.tensor([x, y, z], requires_grad=True).cuda()
    q = kornia.geometry.conversions.rotation_matrix_to_quaternion(R)
    return kornia.geometry.conversions.euler_from_quaternion(*q)

def rotation_Matrix2angle_bins(R, bins=360):
    '''
    Convert a rotation matrix to angle bins
    '''
    x_angle, y_angle, z_angle = rotation_Matrix2angles(R)
    # print(x_angle, y_angle, z_angle)
    x = torch.zeros(bins)
    x[int(torch.rad2deg(x_angle))] = 1
    # print(torch.rad2deg(x_angle))
    # print(int(torch.rad2deg(x_angle)))
    # print(x)

    y = torch.zeros(bins)
    y[int(torch.rad2deg(y_angle))] = 1
    z = torch.zeros(bins)
    z[int(torch.rad2deg(z_angle))] = 1
    return torch.stack([x, y, z])

def angle_bins2rotation_Matrix(angles):
    '''
    Convert angle bins to a rotation matrix
    '''
    # print(angles)
    # print(angles[:,0])
    # print(torch.softmax(angles[:,0], dim=1))
    
    values = torch.arange(360).float().cuda()
    # print(torch.softmax(angles[:, 0], dim=1) * values)
    # print(torch.sum(torch.softmax(angles[:, 0], dim=1) * values, dim=1))
    x = torch.deg2rad(torch.sum(torch.softmax(angles[:, 0], dim=1) * values, dim=1))
    y = torch.deg2rad(torch.sum(torch.softmax(angles[:, 1], dim=1) * values, dim=1))
    z = torch.deg2rad(torch.sum(torch.softmax(angles[:, 2], dim=1) * values, dim=1))
    # print(x, y, z)
    q = kornia.geometry.conversions.quaternion_from_euler(x, y, z)
    # print(q)
    return kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.stack(q).T)

def GS_transform(preds):
    '''
    Calculate the transformation matrix from the predicted vectors
    '''
    pred_z, pred_y = preds
    z = pred_z
    norm_z = torch.linalg.norm(z, axis=1).repeat(3, 1).T
    z = z / norm_z

    y = pred_y
    # y = y - torch.dot(z, y)*z
    y = y - torch.sum(z * y, dim=-1, keepdim=True) * z
    norm_y = torch.linalg.norm(y, axis=1).repeat(3, 1).T
    y = y / norm_y

    x = torch.linalg.cross(y, z)

    transform = torch.zeros([pred_y.shape[0], 3, 3])
    transform[:, :3, 0] = x
    transform[:, :3, 1] = y
    transform[:, :3, 2] = z

    return transform

def quaternion2Rotation_Matrix(quaternion):
    '''
    Convert a quaternion to a rotation matrix
    '''
    return kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)

def rotation_Matrix2quaternion(R):
    '''
    Convert a rotation matrix to a quaternion
    '''
    return kornia.geometry.conversions.rotation_matrix_to_quaternion(R)


class Loss_Calculator:
    def __init__(self, args):
        self.loss_type = args.loss_type

        self.loss_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
        
        self.l2loss_f = torch.nn.MSELoss()
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
    
    def save_results(self, epoch=""):
        np.set_printoptions(suppress=True)
        np.savetxt(f'{self.train_file}{epoch}.out', self.train_loss_all, delimiter=',')
        np.savetxt(f'{self.val_file}{epoch}.out', self.val_loss_all, delimiter=',')
    
    def load_results(self, epoch=""):
        self.train_loss_all = list(np.loadtxt(f'{self.train_file}{epoch}.out', delimiter=','))
        self.val_loss_all = list(np.loadtxt(f'{self.val_file}{epoch}.out', delimiter=','))

    def print_results(self, e, epochs):
        print(20 * "*")
        print("Epoch {}/{}".format(e, epochs))
        print("mean - val loss: {}".format(np.mean(self.val_losses)))
        print("median - val loss: {}".format(np.median(self.val_losses)))

    def is_best(self):
        if min(self.val_loss_all) == self.val_loss_all[-1]:
            return True
        return False
        

class GS_Loss_Calculator(Loss_Calculator):
    def __init__(self, args):
        super().__init__(args)
        self.loss_z_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
        self.loss_y_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
        
        self.val_losses_z = []
        self.val_losses_y = []

        self.function_dict = {'angle_rotmat': (self.calculate_angle_rotmat_loss, 
                                    self.calculate_val_angle_rotmat_loss),
                        'angle_vectors': (self.calculate_angle_vectors_train_loss,
                                    self.calculate_angle_vectors_val_loss),
                        'elements': (self.calculate_elements_loss,
                                    self.calculate_val_elements_loss)
        }
        self.folder = f'/home/kocurvik/bcpravdova/siet_for_bins/training_data_synth/results/GS/{args.loss_type}/'
        self.train_file = self.folder + 'train_err'
        self.val_file = self.folder + 'val_err'
      
    def calculate_elements_loss(self, preds, true_transform):
        pred_z, pred_y = preds
        loss_z = self.l2loss_f(pred_z, true_transform[:, :3, 2].float().cuda())
        loss_y = self.l2loss_f(pred_y, true_transform[:, :3, 1].float().cuda())
        loss = loss_z + loss_y

        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss

    def calculate_val_elements_loss(self, preds, true_transform):
        self.val_losses.append(self.calculate_elements_loss(preds, true_transform).detach().item())

    def calculate_angle_rotmat_loss(self, preds, true_transform):
        true_transfrm = true_transform.float()
        transform = GS_transform(preds)
        loss = calculate_eRE(true_transfrm, transform.cuda()).cuda()

        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss
    
    def calculate_val_angle_rotmat_loss(self, preds, true_transform):
        self.val_losses.append(self.calculate_angle_rotmat_loss(preds, true_transform).detach().item())

    def reset_val_loss(self):
        self.val_losses = []
        self.val_losses_z = []
        self.val_losses_y = []

    def calculate_angle_vectors_train_loss(self, preds, transform): 
        pred_z, pred_y = preds
        loss_z = torch.mean(get_angles(pred_z, transform[:, :3, 2].cuda()))
        loss_y = torch.mean(get_angles(pred_y, transform[:, :3, 1].cuda()))
        loss = loss_z + loss_y  

        self.loss_z_running = 0.9 * self.loss_z_running + 0.1 * loss_z
        self.loss_y_running = 0.9 * self.loss_y_running + 0.1 * loss_y
        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        self.print_running_loss()
        return loss
    
    def calculate_angle_vectors_val_loss(self, preds, transform): 
        pred_z, pred_y = preds
        loss_z = torch.mean(get_angles(pred_z, transform[:, :3, 2].cuda()))
        loss_y = torch.mean(get_angles(pred_y, transform[:, :3, 1].cuda(), sym_inv=True))
        loss = loss_z + loss_y 

        self.val_losses.append(loss.detach().item())
        self.val_losses_z.append(loss_z.detach().item())
        self.val_losses_y.append(loss_y.detach().item())


    # def calculate_angle_vectors_train_loss(self, preds, transform, translation): 
    #     pred_z, pred_y, preds_t = preds
    #     loss_z = torch.mean(get_angles(pred_z, transform[:, :3, 2].cuda()))
    #     loss_y = torch.mean(get_angles(pred_y, transform[:, :3, 1].cuda()))
    #     loss_t = self.l2loss_f(preds_t, translation.float().cuda())
    #     loss = loss_z + loss_y + loss_t

    #     self.loss_z_running = 0.9 * self.loss_z_running + 0.1 * loss_z
    #     self.loss_y_running = 0.9 * self.loss_y_running + 0.1 * loss_y
    #     self.loss_running = 0.9 * self.loss_running + 0.1 * loss

    #     self.print_running_loss()
    #     return loss

    # def calculate_angle_vectors_val_loss(self, preds, transform, translation):
    #     pred_z, pred_y, preds_t = preds
    #     loss_z = torch.mean(get_angles(pred_z, transform[:, :3, 2].cuda()))
    #     loss_y = torch.mean(get_angles(pred_y, transform[:, :3, 1].cuda()))
    #     loss_t = self.l2loss_f(preds_t, translation.float().cuda())
    #     loss = loss_z + loss_y + loss_t

    #     self.val_losses.append(loss.detach().item())
    #     self.val_losses_z.append(loss_z.detach().item())
    #     self.val_losses_y.append(loss_y.detach().item())
        
    

    def print_running_loss(self):
        print("Running loss: {}, z loss: {}, y loss: {}"
            .format(self.loss_running.item(),  self.loss_z_running.item(), self.loss_y_running.item()))
    
    def print_results(self, e, epochs):
        print(20 * "*")
        print("Epoch {}/{}".format(e, epochs))
        print("means - \t val loss: {} \t z loss: {} \t y loss: {}"
                .format(np.mean(self.val_losses), np.mean(self.val_losses_z), np.mean(self.val_losses_y)))
        print("medians - \t val loss: {} \t z loss: {} \t y loss: {} "
                .format(np.median(self.val_losses), np.median(self.val_losses_z), np.median(self.val_losses_y)))


   


class Axis_angle_3D_Loss_Calculator(Loss_Calculator):
    def __init__(self, args):
        super().__init__(args)

        self.function_dict = {'angle_rotmat': (self.calculate_angle_rotmat_loss, 
                                    self.calculate_val_angle_rotmat_loss),
                            'angle_vectors': (self.calculate_angle_vectors_loss,
                                    self.calculate_angle_vectors_val_loss),
                            'elements': (self.calculate_elements_loss,
                                    self.calculate_val_elements_loss)

        }

        self.folder = f'/home/kocurvik/bcpravdova/siet_for_bins/training_data_synth/results/Axis_Angle_3D/{args.loss_type}/'
        self.train_file = self.folder + 'train_err'
        self.val_file = self.folder + 'val_err'
    
    def calculate_angle_rotmat_loss(self, pred, gt):
        pred_R = kornia.geometry.conversions.axis_angle_to_rotation_matrix(pred)
        true_transfrm = gt.float()

        loss = calculate_eRE(true_transfrm, pred_R.cuda()).cuda()
        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss
    
    def calculate_val_angle_rotmat_loss(self, pred, gt):
        self.val_losses.append(self.calculate_angle_rotmat_loss(pred, gt).detach().item())

    def calculate_angle_vectors_loss(self, preds, true_transform):
        gt_axis = kornia.geometry.conversions.rotation_matrix_to_axis_angle(true_transform)
        gt_angle = torch.linalg.norm(gt_axis, axis=1)

        preds_angle = torch.linalg.norm(preds, axis=1)

        loss_vec = torch.mean(get_angles(preds, gt_axis.cuda()))
        loss_angle = self.l2loss_f(preds_angle, gt_angle.float().cuda())

        loss = loss_vec + loss_angle

        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss
    
    def calculate_angle_vectors_val_loss(self, preds, true_transform):
        loss = self.calculate_angle_vectors_loss(preds, true_transform)
        self.val_losses.append(loss.detach().item())

    def calculate_elements_loss(self, preds, true_transform):
        gt_axis = torch.stack([kornia.geometry.conversions.rotation_matrix_to_axis_angle(true_transform[i]) for i in range(len(true_transform))])
        loss = self.l2loss_f(preds, gt_axis.float().cuda())
        # loss = torch.mean(torch.abs(preds - gt_axis)).cuda()

        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss
    
    def calculate_val_elements_loss(self, preds, true_transform):
        loss = self.calculate_elements_loss(preds, true_transform)
        self.val_losses.append(loss.detach().item())



class Axis_angle_4D_Loss_Calculator(Loss_Calculator):
    def __init__(self, args):
        super().__init__(args)

        self.function_dict = {'angle_rotmat': (self.calculate_angle_rotmat_loss, 
                                    self.calculate_val_angle_rotmat_loss),
                        'elements': (self.calculate_elements_loss,
                                    self.calculate_val_elements_loss),
                        'angle_vectors': (self.calculate_angle_vectors_loss,
                                    self.calculate_val_angle_vectors_loss)
                            }

        self.folder = f'/home/kocurvik/bcpravdova/siet_for_bins/training_data_synth/results/Axis_Angle_4D/{args.loss_type}/'
        self.train_file = self.folder + 'train_err'
        self.val_file = self.folder + 'val_err'

    def calculate_angle_rotmat_loss(self, pred, gt):
        axis, angle = pred
        axis = axis / torch.linalg.norm(axis, axis=1).unsqueeze(1)
        
        pred_R = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis*angle)
        true_transfrm = gt.float()

        loss = calculate_eRE(true_transfrm, pred_R.cuda()).cuda()
        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss
    
    def calculate_val_angle_rotmat_loss(self, pred, gt):
        self.val_losses.append(self.calculate_angle_rotmat_loss(pred, gt).detach().item())

    def calculate_elements_loss(self, preds, true_transform):
        gt_axis = torch.stack([kornia.geometry.conversions.rotation_matrix_to_axis_angle(true_transform[i]) for i in range(len(true_transform))])
        gt_angle = torch.linalg.norm(gt_axis, axis=1)
        gt_axis = gt_axis / gt_angle.unsqueeze(1)
        gt = torch.cat([gt_axis, gt_angle.unsqueeze(1)], dim=1)

        loss = self.l2loss_f(torch.cat(preds, dim=1), gt.float().cuda())

        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss
    
    def calculate_val_elements_loss(self, preds, true_transform):
        loss = self.calculate_elements_loss(preds, true_transform)
        self.val_losses.append(loss.detach().item())

    def calculate_angle_vectors_loss(self, preds, true_transform):
        gt_axis = kornia.geometry.conversions.rotation_matrix_to_axis_angle(true_transform)
        gt_angle = torch.linalg.norm(gt_axis, axis=1)
        gt_axis = gt_axis / gt_angle.unsqueeze(1)

        loss_vec = torch.mean(get_angles(preds[0], gt_axis.cuda()))
        loss_angle = self.l2loss_f(preds[1], gt_angle.float().cuda())
        loss = loss_vec + loss_angle

        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss
    
    def calculate_val_angle_vectors_loss(self, preds, true_transform):
        loss = self.calculate_angle_vectors_loss(preds, true_transform)
        self.val_losses.append(loss.detach().item())


class Stereographic_Loss_Calculator(Loss_Calculator):
    def __init__(self, args):
        super().__init__(args)

        self.function_dict = {'angle_rotmat': (self.calculate_angle_rotmat_loss, 
                                    self.calculate_val_angle_rotmat_loss),
                        'angle_vectors': (self.calculate_angle_vectors_loss,
                                    self.calculate_val_angle_vectors_loss),
        }

        self.folder = f'/home/kocurvik/bcpravdova/siet_for_bins/training_data_synth/results/Stereographic/{args.loss_type}/'
        self.train_file = self.folder + 'train_err'
        self.val_file = self.folder + 'val_err'

    def get_transform(self, pred):
        first, last = pred[:,:2], pred[:,2:]
        norm_last = torch.linalg.norm(last, axis=1)
        # print(norm_last)
        new_element = (norm_last**2 - 1) /2 
        # print(new_element)
        new_last = torch.stack([new_element/norm_last, last[:, 0]/norm_last, last[:, 1]/norm_last, last[:, 2]/norm_last], dim=1)
        gs = torch.cat([first, new_last], dim=1)
        # print(gs)
        gs = gs.reshape(-1, 2, 3).transpose(1,0)
        # print(gs)
        return gs
        

    def calculate_angle_rotmat_loss(self, pred, gt):
        transform = self.get_transform(pred)
        transform = GS_transform(transform)

        true_transfrm = gt.float()
        loss = calculate_eRE(true_transfrm, transform.cuda()).cuda()

        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss
    

    def calculate_val_angle_rotmat_loss(self, pred, gt):
        self.val_losses.append(self.calculate_angle_rotmat_loss(pred, gt).detach().item())

    def calculate_angle_vectors_loss(self, pred, gt):
        transform = self.get_transform(pred)
        pred_z, pred_y = transform[0], transform[1]
        gt = gt.float()

        loss_z = torch.mean(get_angles(pred_z,  gt[:, :3, 2].cuda()))
        loss_y = torch.mean(get_angles(pred_y,  gt[:, :3, 1].cuda()))
        loss = loss_z + loss_y

        self.loss_running = 0.9 * self.loss_running + 0.1 * loss

        print(f'Running Loss: {self.loss_running.item()}')
        return loss

    def calculate_val_angle_vectors_loss(self, pred, gt):
        self.val_losses.append(self.calculate_angle_vectors_loss(pred, gt).detach().item())
        
