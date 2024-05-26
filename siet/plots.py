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
    plt.savefig(f'images/Losses_{title}.pdf')
    plt.show()

def roc_curve(paths, dataset, name):
    plt.clf()
    i=0
    for path in paths:
        i, rep, loss = path
        pth = os.path.join('siet', 'training_data', dataset, 'results',rep, loss, 'err_by_index.csv')
        if not os.path.exists(pth):
            print(f'Path {pth} does not exist')
            continue
        errs = np.loadtxt(pth, delimiter=',')[:, 1]
        curve = np.array([np.sum(errs < t) / len(errs) for t in range(11)])
        # i+=1
        plt.plot(curve, label=f'{i:2}  {rep:<20} {loss}', color=colors[i-1])
        
    plt.xlabel('Veľkosť chyby angle_rotmat x[°]')
    plt.ylabel('Podiel vzoriek s chybou menšou ako x')
    # plt.xticks(range(0, 91, 10), [str(i) for i in range(0, 91, 10)])
    # logaritmicke osi
    # plt.xscale('log')
    plt.title(name, fontsize=18)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
    plt.savefig(f'images/vyberROC_{dataset}.pdf', bbox_inches='tight')
    plt.show()
    

def closest_vs_errors(cube, paths, name, cmap):
    plt.clf()
    path = os.path.join('datasets', cube, 'closest.csv')   
    closest = np.loadtxt(path, delimiter=',')[:, 2]
    labels = []
    i=0
    for path in paths:
        i, repre, loss = path
        pth = os.path.join('siet', 'training_data', cube, 'results', repre, loss, 'err_by_index.csv')
        if not os.path.exists(pth):
            print(f'Path {pth} does not exist')
            continue
        errs = np.loadtxt(pth, delimiter=',')[:, 1]
        # i+=1
        plt.scatter(closest, errs, s=0.5, label=f'{i:2}   {repre:<15} {loss}', color=cmap(i-1))
        a, b = np.polyfit(closest, errs, 1)
        print(f'{repre} {loss} {a} {b}')
        plt.plot(closest, a*closest+b, color=cmap(i-1), linewidth=1.5)
        labels.append(f'{i:2}  {repre:<14} {loss}\n\n    y = {a:.2f}x + {b:.2f}')

    plt.xlabel('Uhol k najbližšej matici v trénovacej množine [°]')
    plt.ylabel('Výsledná chyba angle_rotmat [°]')
    plt.title(name, fontsize=18)
    plt.xlim(0, min(50, max(closest)))
    plt.ylim(0, min(30, max(errs)))
    handles = [Patch(color=cmap(int(l.split()[0])-1), label=labels[i]) for i ,l in enumerate(labels)]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., labelspacing=1, handleheight=5.5,  fontsize=11, prop = {'family': 'monospace'})

    plt.savefig(f'images/Closest_vs_errors_{cube}_all.pdf', bbox_inches='tight')
    plt.show()


def lala():
    closest_priem = np.array([6.44, 9.88, 18.99])
    closest_med = np.array([6.3, 9.85, 17.5])
    closest = [closest_priem, closest_med]
    res = np.array( [ [ [ 1.99, 1.87 ], [ 2.24, 2.05 ], [ 5.90, 4.81]],
    [ [ 2.26, 2.13 ], [ 2.49, 2.08 ], [ 5.97, 5.04]],
    [ [ 2.29, 2.13 ], [ 2.61, 2.44 ], [ 7.49, 5.43]],
    [ [ 2.41, 2.21 ], [ 2.60, 2.45 ], [ 7.94, 7.30]],
    [ [ 3.46, 2.25 ], [ 2.86, 2.55 ], [ 8.00, 6.43]]])

    values_priem = res[:, :, 0]
    values_med = res[:, :, 1]

    colors = generate_colormap_colors(5, "cool_r")
    cmap = mcolors.ListedColormap(colors)

    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    for i in range(4, -1, -1):
        ax[0].scatter(closest_priem, values_priem[i], s=100, label=f'{i+1}.', color=colors[i])
        ax[1].scatter(closest_med, values_med[i], s=100, color=colors[i])

    ax[0].set_xlabel('Priemerná chyba najlepších metód [°]')
    ax[0].set_ylabel('Priemerný uhol ku najbližšej matici\nv trénovacej množine [°]')
    ax[0].set_title('Priemer')
    # a,b = np.polyfit(closest_priem.repeat(5), np.log(values_priem.T.flatten()), 1)
    # lspc = np.linspace(5, 20, 100)
    # ax[0].plot(lspc, np.exp(a*lspc)+ b , '--', color='gray', linewidth=1.5)
    
    ax[1].set_xlabel('Mediánová chyba najlepších metód [°]')
    ax[1].set_ylabel('Mediánový uhol ku najbližšej matici\nv trénovacej množine ζ [°]')
    ax[1].set_title('Medián')
    # a, b = np.polyfit(closest_med.repeat(5), np.log(values_med.T.flatten()), 1)
    # lspc = np.linspace(5, 20, 100)
    # ax[1].plot(lspc, np.exp(a*lspc)+b, '--', color='gray', linewidth=1.5)
    # print the coeeficients a b in the picture
    # ax[0].text(0.3, 0.9, f'y = exp({a:.2f}x) + {b:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
    
    # plt.savefig('images/Closest_vs_errors.png')
    fig.suptitle('Vzťah medzi uhlom k najbližšej matici v trénovacej množine a výslednou chybou', fontsize=14)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.01), ncol=5, title='Poradie metódy')
    plt.tight_layout()
    plt.savefig('images/Closest_vs_errors.pdf', bbox_inches='tight')
    plt.show()


def generate_colormap_colors(n, cmap_name):
    """Generate n distinct, evenly spaced colors using the specified colormap."""
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / n) for i in range(n)]
    return colors

if __name__ == "__main__":
    reprs = [
        'Euler',
        'Euler_binned',
        'Quaternion',
        'Axis_Angle_3D',
        'Axis_Angle_4D',
        # 'Axis_Angle_binned',
        'Stereographic',
        'GS',
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
    num_colors = 20
    # colors = generate_colormap_colors(num_colors, "gist_ncar")
   

    colors = [
    '#8F2323',
    # "#FF0000",
    "#FF7F00", 
    # "#FFD400",
    "#ffe119", 
    # "#BFFF00", 
    '#88DDD5',
    # "#6ACC55", 
    # "#04a82a",
    '#4699BB',
    # '#00EAFF',
    # "#0095FF",
    # "#0040FF",
    # '#4e1bf9',
    "#8855FF",
    # '#aaaaff',
    # '#e47bfc',
    "#fabed4", 
    # '#FF00AA',
    # "#98327d", 


    
    
]
    cmap = mcolors.ListedColormap(colors)
    # plt.style.use('seaborn-v0_8-whitegrid')
    # change font for all text
    # plt.rcParams.update({'font.size': 14, 'font.family': 'monospace'})
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors)

    zaujimave_paths = [
                        (9, 'Axis_Angle_3D', 'angle_vectors'),
                        (11, 'Axis_Angle_4D', 'elements'),
                        (15, 'Stereographic', 'angle_rotmat'),
                        (18, 'GS', 'angle_rotmat'),
                       (19, 'GS', 'elements'),
                       (20, 'GS', 'angle_vectors'),
                       ]




    # path = os.path.join('siet', 'training_data', dataset, 'results', repre, lossf)
    # plot_losses(path, 'train_err.out', 'val_err.out',  f'{repre} {lossf}')
    # paths = [ (repre, lossf)  for repre in reprs for lossf in losses]
    # paths_n_names = zip( datasets, ['Náhodný dataset', 'Dataset s veľkou dierou', "Dataset s veľa dierami", "Pestrofarebný dataset", "Jednofarebný dataset"])
    # for dataset, m in zip( datasets, ['Náhodný dataset', 'Dataset s veľkou dierou', "Dataset s veľa dierami", "Pestrofarebný dataset", "Jednofarebný dataset"]):
    #     roc_curve(zaujimave_paths, dataset, m)
    # # closest_vs_errors('cube_cool', paths, cmap)
    # for dataset, m in paths_n_names:
    #     closest_vs_errors(dataset, zaujimave_paths, m, cmap)
    fig, ax = plt.subplots(1, figsize=(8,5))
    loss = 'angle_rotmat'
    for i, repre in enumerate(reprs):
        path = os.path.join('siet', 'training_data', 'cube_colorful', 'results', repre, loss, 'train_err.out')
        if not os.path.exists(path):
            print(f'Path {path} does not exist')
            continue
        with open(path, 'r') as f:
            train_lines = f.readlines()
            train_lines = [float(line.strip()) for line in train_lines]
        
        path = os.path.join('siet', 'training_data', 'cube_colorful', 'results', repre, loss, 'val_err.out')
        if not os.path.exists(path):
            print(f'Path {path} does not exist')
            continue

        with open(path, 'r') as f:
            val_lines = f.readlines()
            val_lines = [float(line.strip()) for line in val_lines]
        ax.plot(train_lines, '--', label=f'{repre} train', color = colors[i], alpha=0.7)
        ax.plot(val_lines,  label=f'{repre} val', color = colors[i], alpha=1)
    ax.set_title(f'angle_rotmat')
    # (lines, labels) = plt.gca().get_legend_handles_labels()
    # leg1 = plt.legend(lines[:1], labels[:1], bbox_to_anchor=(0,0,0.8,1), loc=1)
    # leg2 = plt.legend(lines[1:], labels[1:], bbox_to_anchor=(0,0,1,1), loc=1)
    # gca().add_artist(leg1)
    plt.legend(ncol=2)
    ax.set_xlabel('Epocha')
    ax.set_ylabel('Chyba')
    ax.set_ylim(0, 30)
    plt.savefig(f'images/Losses_angle_rotmat.pdf', bbox_inches='tight')
    plt.show()
