import numpy as np
import argparse
import os

from prettytable import PrettyTable
from dataset import Dataset
from torch.utils.data import DataLoader
from network import parse_command_line


def angle(x, y):
    x = x.ravel()
    y = y.ravel()
    return np.rad2deg(np.arccos(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))))

def rotation_angle(R):
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

def calculate_eRE(R_gt, R_est):
    R = R_gt.T @ R_est    
    return rotation_angle(R)


def evaluate(truth, pred, path_res):
    true_matrices = np.loadtxt(truth, delimiter=',')
    pred_matrices = np.loadtxt(pred, delimiter=',')

    eRE_list = []

    counter_better = 0
    counter_worse = 0
    with open(path_res, 'w') as f:
        for i, row in enumerate(pred_matrices):
            # print(f'Index: {index}')
            R_gt = true_matrices[i].reshape(3, 3)
            # print(f'GT: {R_gt}')
            R_est = row.reshape(3, 3)
            # print(f'EST: {R_est}')
            err = calculate_eRE(R_gt, R_est)
            # print(f'eRE: {err}')
            eRE_list.append(err)
            if err > 10:
                counter_worse += 1
            else:
                counter_better += 1

            print(f'{i},{err}', file=f)

    return eRE_list

def save_results(results, repre, loss_f, dataset):
    tab = PrettyTable(['metric', 'dataset', 'represetnation', 'loss_function', 'epoch', 'median', 'mean', '5', '10', 'all'])
    tab.align["metric"] = "l"
    tab.float_format = '0.2'
    for res in results:
        errs = np.array(res['eRE'])
        errs[np.isnan(errs)] = 180
        result = np.array([np.sum(errs < t) / len(errs) for t in range(1, 21)])
        tab.add_row(['eRE', dataset, repre, loss_f, res['epoch'], np.median(errs), np.mean(errs), np.mean(result[:5]), np.mean(result[:10]), np.mean(result)])

    print(tab)

    with open(f'siet_for_bins/training_data_synth/results/{repre}/{loss_f}/bestprettytab_results.csv', 'w') as f:
        f.write(tab.get_string())
    with open(f'siet_for_bins/training_data_synth/results/{repre}/{loss_f}/bestcsv_results.csv', 'w') as f:
        f.write(tab.get_csv_string())


if __name__ == '__main__':
    """
    Example usage: python evaluate.py path/to/dataset_with_predictions
    """
    truth = 'siet_for_bins/test_synth_matrices.csv'

    for repre, loss_type in [('GS', 'elements'), ('GS', 'angle_rotmat'), ('GS', 'angle_vectors'), ('Axis_Angle_4D', 'elements'), ['Stereographic', 'angle_rotmat']]:
        # loss_type = 'elements'
        path = f'siet_for_bins/training_data_synth/results/{repre}/{loss_type}/'

        results = []
        pred = f'siet_for_bins/training_data_synth/inferences/{repre}/{loss_type}/infer_results_best.csv'
        if not os.path.exists(pred):
            print(f'Path {pred} does not exist')
            continue

        eRE_list = evaluate(truth, pred, f'{path}besterr_by_index.csv')
        dic = {'eRE': eRE_list,
            'epoch': 'best'  }
        results.append(dic)

        save_results(results, repre, loss_type, 'VISIGRAPP_TEST')

        # for i in range(0, 201, 10):
        #     pred = f'siet_for_bins/training_data_synth/inferences/{repre}/{loss_type}/infer_results{i:03d}.csv'
        #     if not os.path.exists(pred):
        #         print(f'Path {pred} does not exist')
        #         break
            
        #     eRE_list = evaluate(truth, pred, f'{path}err_by_index.csv')
        #     dic = {'eRE': eRE_list,
        #         'epoch': i  }
        #     results.append(dic)
            
        #     with open(f'{path}test_err_by_epochs.csv', 'a') as f:
        #         print(f'{i},{np.mean(eRE_list)},{np.median(eRE_list)},{np.max(eRE_list)},{np.min(eRE_list)},{np.std(eRE_list)},{np.percentile(eRE_list, 90)}', file=f)
        
        # save_results(results, repre, loss_type, 'VISIGRAPP_TEST') 
