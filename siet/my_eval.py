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


def evaluate(truth, pred, path_res):
    true_matrices = np.loadtxt(truth, delimiter=',')
    pred_matrices = np.loadtxt(pred, delimiter=',')

    eRE_list = []

    counter_better = 0
    counter_worse = 0
    with open(path_res, 'w') as f:
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

            print(f'{index},{err}', file=f)

    # print(eRE_list)
    # print(len(eRE_list))
    # print(f'better: {counter_better}')
    # print(f'worse: {counter_worse}')
    # print(f'Mean eRE: {np.mean(eRE_list)}')
    # print(f'Median eRE: {np.median(eRE_list)}')
    # print(f'Max eRE: {np.max(eRE_list)}')
    # print(f'Min eRE: {np.min(eRE_list)}')
    # print(f'Std eRE: {np.std(eRE_list)}')
    # print(f'90th percentile eRE: {np.percentile(eRE_list, 90)}')

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

    with open(f'siet/training_data/{dataset}/results/{repre}/{loss_f}/prettytab_results.csv', 'w') as f:
        f.write(tab.get_string())
    with open(f'siet/training_data/{dataset}/results/{repre}/{loss_f}/csv_results.csv', 'w') as f:
        f.write(tab.get_csv_string())


if __name__ == '__main__':
    """
    Example usage: python evaluate.py path/to/dataset_with_predictions
    """
    reprs = [
        'GS',
        'Euler',
        'Euler_binned',
        'Quaternion',
        'Axis_Angle_3D',
        'Axis_Angle_4D',
        'Axis_Angle_binned',
        'Stereographic',
        # 'Matrix'
    ]
    losses = [
            'angle_rotmat',
            'elements',
            'angle_vectors'
              ]

    datasets = [
            'cube_cool', 
            'cube_big_hole', 
            'cube_dotted', 
            'cube_colorful', 
            'cube_one_color'
            ]

    for dset in datasets:
        for repre in reprs:
            for loss_type in losses:
                path_pics = f'datasets/{dset}'

                truth = f'{path_pics}/test/matice.csv'
               
                path = f'siet/training_data/{dset}/results/{repre}/{loss_type}/'
                if os.path.exists(f'{path}test_err_by_epochs.csv'):
                    os.remove(f'{path}test_err_by_epochs.csv')

                if not os.path.exists(path):
                    print(f'Path {path} does not exist')
                    break 
                
                results = []
                for i in range(0, 101, 10):
                    pred = f'siet/training_data/{dset}/inferences/{repre}/{loss_type}/infer_results{i:03d}.csv'
                    if not os.path.exists(pred):
                        print(f'Path {pred} does not exist')
                        break
                    eRE_list = evaluate(truth, pred, f'{path}err_by_index.csv')
                    dic = {'eRE': eRE_list,
                        'epoch': i  }
                    results.append(dic)
                    
                    with open(f'{path}test_err_by_epochs.csv', 'a') as f:
                        print(f'{i},{np.mean(eRE_list)},{np.median(eRE_list)},{np.max(eRE_list)},{np.min(eRE_list)},{np.std(eRE_list)},{np.percentile(eRE_list, 90)}', file=f)
                
                save_results(results, repre, loss_type, dset) 
