from matplotlib import pyplot as plt
import numpy as np


from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'

def sphere(path, filt, col):

    table = np.loadtxt(path, delimiter = ',')
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    r = 0.99
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

    ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='gray', alpha=0.1, linewidth=0)

    for i, *x in table:
        i = int(i) 
        vec = np.array([1, 0, 0])
        R = np.array(x).reshape(3, 3)
        vec = R @ vec
        color = col[filt[i%10]]
        # mlab.points3d(vec[0], vec[1], vec[2], color = color, scale_factor = 0.05)

        ax.scatter([0, vec[0]], [0, vec[1]], [0, vec[2]], c = color, s = 1, alpha = 1)
    
    # mlab.savefig('sphere.png')

    plt.savefig('sphere.png')

if __name__ == '__main__':
    path = 'matice.csv'
    # if %10 is 0, then val, if %10 is 5, then test, else train
    filt = {0: 'val', 5: 'test', 1: 'train', 2: 'train', 3: 'train', 4: 'train', 6: 'train', 7: 'train', 8: 'train', 9: 'train'}
    # col = {'train': (0,0,1), 'val': (0,1,0), 'test': (1,0,0)}
    col = {'train': 'lightblue', 'val': 'green', 'test': 'red'}
    sphere(path, filt, col)