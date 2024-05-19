from matplotlib import pyplot as plt
import numpy as np
import os


def plot_losses(path, train, val, title):
    with open(f'{path}/{train}', 'r') as f:
        train_lines = f.readlines()
        train_lines = [float(line.strip()) for line in train_lines]

    with open(f'{path}/{val}', 'r') as f:
        val_lines = f.readlines()
        val_lines = [float(line.strip()) for line in val_lines]
        plt.plot(train_lines, label='train')
        plt.plot(val_lines, label='val')

    plt.title(title)
    plt.legend()
    plt.savefig(f'Losses_{title}.png')
    plt.show()

def roc_curve(paths, dataset):
    plt.clf()
    for path in paths:
        try:
            pth = os.path.join('siet', 'training_data', dataset, 'results',path)
            errs = np.loadtxt(f'{pth}/err_by_index.csv', delimiter=',')[:, 1]
            curve = np.array([np.sum(errs < t) / len(errs) for t in range(1, 181)])
            plt.plot(curve, label=path)
        except:
            print(f'Error in {path}')
            continue
    plt.title(dataset)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
    plt.savefig(f'ROC_{dataset}.png', bbox_inches='tight')
    plt.show()
    

def closest_vs_errors(path, title):
    pass
    
if __name__ == "__main__":

    reprs = [
        'GS',
        'Euler',
        'Euler_binned',
        'Quaternion',
        'Axis_Angle_3D',
        'Axis_Angle_4D',
        'Axis_Angle_binned',
        # 'Stereographic',
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
    # path = os.path.join('siet', 'training_data', dataset, 'results', repre, lossf)
    # plot_losses(path, 'train_err.out', 'val_err.out',  f'{repre} {lossf}')
    paths = [ f'{repre}/{lossf}'  for repre in reprs for lossf in losses]
    for dataset in datasets:
        roc_curve(paths, dataset)
