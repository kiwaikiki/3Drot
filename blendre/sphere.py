from matplotlib import pyplot as plt
import numpy as np


def sphere(path, filt, col):
    table = np.loadtxt(path, delimiter = ',')
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    for i, *x in table:
        i = int(i) 
        vec = np.array([1, 1, 1])
        R = np.array(x).reshape(3, 3)
        vec = R @ vec
        color = col[filt[i%10]]
        ax.scatter([0, vec[0]], [0, vec[1]], [0, vec[2]], c = color, s = 1)  
        
    plt.savefig('sphere.png')

if __name__ == '__main__':
    path = 'matice.csv'
    # if %10 is 0, then val, if %10 is 5, then test, else train
    filt = {0: 'val', 5: 'test', 1: 'train', 2: 'train', 3: 'train', 4: 'train', 6: 'train', 7: 'train', 8: 'train', 9: 'train'}
    col = {'train': 'lightblue', 'val': 'g', 'test': 'r'}
    sphere(path, filt, col)