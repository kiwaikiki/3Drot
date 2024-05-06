import numpy as np
import argparse
import os

from my_loss import angles2Rotation_Matrix
# import PrettyTable

def angle(x, y):
    x = x.ravel()
    y = y.ravel()
    return np.rad2deg(np.arccos(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))))

def rotation_angle(R):
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

def calculate_eTE(R_gt, R_est):
    R = R_gt.T @ R_est    
    return rotation_angle(R)


def evaluate(truth, pred):
    true_matrices = np.loadtxt(truth, delimiter=',')
    pred_matrices = np.loadtxt(pred, delimiter=',')

    eTE_list = []

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
        err = calculate_eTE(R_gt, R_est)
        # print(f'eTE: {err}')
        eTE_list.append(err)
        if err > 10:
            counter_worse += 1
        else:
            counter_better += 1


    print(len(eTE_list))
    print(counter_better)
    print(counter_worse)
    print(f'Mean eTE: {np.mean(eTE_list)}')
    print(f'Median eTE: {np.median(eTE_list)}')
    print(f'Max eTE: {np.max(eTE_list)}')
    print(f'Min eTE: {np.min(eTE_list)}')
    print(f'Std eTE: {np.std(eTE_list)}')
    print(f'90th percentile eTE: {np.percentile(eTE_list, 90)}')

    return eTE_list

# def print_results(results):
#     tab = PrettyTable(['metric', 'median', 'mean', 'AUC@5', 'AUC@10', 'AUC@20'])
#     tab.align["metric"] = "l"
#     tab.float_format = '0.2'
#     err_names = ['P_12_err', 'P_13_err', 'P_23_err', 'P_err']
#     for err_name in err_names:
#         errs = np.array([r[err_name] for r in results])
#         errs[np.isnan(errs)] = 180
#         res = np.array([np.sum(errs < t) / len(errs) for t in range(1, 21)])
#         tab.add_row([err_name, np.median(errs), np.mean(errs), np.mean(res[:5]), np.mean(res[:10]), np.mean(res)])

if __name__ == '__main__':
    """
    Example usage: python evaluate.py path/to/dataset_with_predictions
    """
    truth = '../blendre/matice.csv'
    path = 'GS/elements/'

    with open(f'results/{path}test_err_by_epochs.csv', 'w') as f:
            pass
    
    for i in range(0, 101, 10):
        pred = f'inferences/{path}infer_results{i:03d}.csv'
        # print(pred)
        if not os.path.exists(pred):
            print(f'Path {pred} does not exist')
            break

        eTE_list = evaluate(truth, pred)

        with open(f'results/{path}test_err_by_epochs.csv', 'a') as f:
            print(f'{i},{np.mean(eTE_list)},{np.median(eTE_list)},{np.max(eTE_list)},{np.min(eTE_list)},{np.std(eTE_list)},{np.percentile(eTE_list, 90)}', file=f)
        # print(f'90th percentile eTE: {np.percentile(eTE_list, 90)}')
        # print(f'Mean eTE: {np.mean(eTE_list)}')
        # print(f'Median eTE: {np.median(eTE_list)}')
        # print(f'Max eTE: {np.max(eTE_list)}')
        # print(f'Min eTE: {np.min(eTE_list)}')
        # print(f'Std eTE: {np.std(eTE_list)}')
