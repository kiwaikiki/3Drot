from matplotlib import pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
# plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
 

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
        pth = os.path.join('siet', 'training_data', dataset, 'results',path, 'err_by_index.csv')
        if not os.path.exists(pth):
            print(f'Path {pth} does not exist')
            continue
        errs = np.loadtxt(pth, delimiter=',')[:, 1]
        curve = np.array([np.sum(errs < t) / len(errs) for t in range(1, 181)])
        plt.plot(curve, label=path)
        
    plt.title(dataset)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
    plt.savefig(f'ROC_{dataset}.png', bbox_inches='tight')
    plt.show()
    

def closest_vs_errors(cube, paths, cmap):
    path = os.path.join('datasets', cube, 'closest.csv')   
    closest = np.loadtxt(path, delimiter=',')[:, 2]
    labels = []
    for path in paths:
        pth = os.path.join('siet', 'training_data', cube, 'results', path, 'err_by_index.csv')
        if not os.path.exists(pth):
            print(f'Path {pth} does not exist')
            continue
        errs = np.loadtxt(pth, delimiter=',')[:, 1]
        plt.scatter(closest, errs, s=0.3, label=path, cmap=cmap)
        labels.append(path)

    plt.xlabel('Closest')
    plt.ylabel('Error')
    plt.title(f'{cube} all')
    plt.xlim(0, 50)
    handles = [Patch(color=cmap(i), label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=handles, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'Closest_vs_errors_{cube}_all.png', bbox_inches='tight')
    plt.show()


def generate_colormap_colors(n, cmap_name):
    """Generate n distinct, evenly spaced colors using the specified colormap."""
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / n) for i in range(n)]
    return colors

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
    num_colors = 15
    colors = generate_colormap_colors(num_colors, "gist_ncar")

    cmap = mcolors.ListedColormap(colors)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors)


    # path = os.path.join('siet', 'training_data', dataset, 'results', repre, lossf)
    # plot_losses(path, 'train_err.out', 'val_err.out',  f'{repre} {lossf}')
    paths = [ f'{repre}/{lossf}'  for repre in reprs for lossf in losses]
    # for dataset in datasets:
    roc_curve(paths, 'cube_big_hole')
  
    closest_vs_errors('cube_big_hole', paths, cmap)