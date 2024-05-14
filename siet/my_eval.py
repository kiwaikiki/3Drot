import numpy as np
import argparse
import os

from prettytable import PrettyTable

from my_loss import angles2Rotation_Matrix
# import PrettyTable

def angle(x, y):
    x = x.ravel()
    y = y.ravel()
    return np.rad2deg(np.arccos(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))))

def rotation_angle(R):
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

def calculate_eRE(R_gt, R_est):
    R = R_gt.T @ R_est    
    return rotation_angle(R)


def evaluate(truth, pred):
    true_matrices = np.loadtxt(truth, delimiter=',')
    pred_matrices = np.loadtxt(pred, delimiter=',')

    eRE_list = []

    counter_better = 0
    counter_worse = 0

    for row in pred_matrices:
        # print(row)
        # print(true_matrices[int(row[0])])
        index = int(row[0])
        # print(f'Index: {index}')
        R_gt = true_matrices[index][1:].reshape(3, 3)
        # print(f'GT: {R_gt}')
        R_est = row[1:].reshape(3, 3)
        # print(f'EST: {R_est}')
        err = calculate_eRE(R_gt, R_est)
        # print(f'eRE: {err}')
        eRE_list.append(err)
        if err > 10:
            counter_worse += 1
        else:
            counter_better += 1

    # print(eRE_list)
    # print(len(eRE_list))
    print(f'better: {counter_better}')
    print(f'worse: {counter_worse}')
    print(f'Mean eRE: {np.mean(eRE_list)}')
    print(f'Median eRE: {np.median(eRE_list)}')
    print(f'Max eRE: {np.max(eRE_list)}')
    print(f'Min eRE: {np.min(eRE_list)}')
    print(f'Std eRE: {np.std(eRE_list)}')
    print(f'90th percentile eRE: {np.percentile(eRE_list, 90)}')

    return eRE_list

def print_results(results):
    tab = PrettyTable(['metric', 'median', 'mean', '5', '10', 'all'])
    tab.align["representation"] = "l"
    tab.float_format = '0.2'
    err_names = ['eRE']
    for err_name in err_names:
        errs = np.array([r[err_name] for r in results])
        # print(errs) 
        errs[np.isnan(errs)] = 180
        res = np.array([np.sum(errs < t) / len(errs) for t in range(1, 21)])
        tab.add_row([err_name, np.median(errs), np.mean(errs), np.mean(res[:5]), np.mean(res[:10]), np.mean(res)])
    print(tab)

if __name__ == '__main__':
    """
    Example usage: python evaluate.py path/to/dataset_with_predictions
    """
    path_pics = 'cool_cube'
    truth = f'{path_pics}/test/matice.csv'
    repre = 'GS'
    loss_used = 'angle'
    path = f'{repre}/{loss_used}/'

    with open(f'siet/training_data/{path_pics}/results/{path}test_err_by_epochs.csv', 'w') as f:
            pass
    
    for i in range(0, 101, 10):
        pred = f'siet/training_data/{path_pics}/inferences/{path}infer_results{i:03d}.csv'
        # print(pred)
        if not os.path.exists(pred):
            print(f'Path {pred} does not exist')
            break

        eRE_list = evaluate(truth, pred)
        dic = {'eRE': eRE_list}
        print_results([dic])

        with open(f'siet/training_data/{path_pics}/results/{path}test_err_by_epochs.csv', 'a') as f:
            print(f'{i},{np.mean(eRE_list)},{np.median(eRE_list)},{np.max(eRE_list)},{np.min(eRE_list)},{np.std(eRE_list)},{np.percentile(eRE_list, 90)}', file=f)
