from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


'''
potrebujem:
bud
1)
- pre kazdy bod vo vygenerovanom datasete najst ci patri do nejakeho blobu alebo nie a olabelovat + najst vzdialenost k najblizsiemu boduv datasete

2)
- vygenerovat dataset s d kruhmi
- vygenerujem bod, checknem ci je v nejakom kruhu, zadlim na test / train
- potom pre kazdy v testovacom prejdem vsetky v trenovaciom a 
    - 
'''

def geodesic_distance(p1, p2, r=1):
    '''
        Calculate geodesic distance between two points on the sphere
        Parameters:
            p1 (np.array): first point
            p2 (np.array): second point
        
        Returns:
            float: geodesic distance between the points
    '''
    dot = min(1, max(-1, np.dot(p1, p2)))
    return r * np.arccos(dot)

def is_in_blob(point, blobs):
    '''
        Check if the point is in any of the blobs
        Parameters:
            point (np.array): point to check
            blobs (list of np.arrays): list of blobs
        
        Returns:
            int: index of the blob if the point is in any of the blobs, -1 otherwise
    '''
    for center, r in blobs:
        if geodesic_distance(point, center) < r:
            return True
    return False

def generate_blobs(n, max_r, min_r):
    '''
        Generate n blobs with radius r and distance d
        Parameters:
            n (int): number of blobs
            r (float): radius of the blobs
            d (float): distance between the blobs
        
        Returns:
            list of np.arrays: list of blobs
    '''
    blobs = []
    for i in range(n):
        center = random_point_on_sphere(1)
        r = np.random.uniform(min_r, max_r)
        # check whether the blob is not too close to the other blobs
        while any([np.linalg.norm(center - c) < r + R + 0.05 for c, R in blobs]):
            center = np.random.rand(3)
            r = np.random.uniform(min_r, max_r)
        blobs.append((center, r))
    return blobs

def save_blobs(blobs, path):    
    with open(path, 'w') as f:
        for center, r in blobs:
            f.write(f'{center[0]},{center[1]},{center[2]},{r}\n')

def load_blobs(path):
    blobs = []
    table = np.loadtxt(path, delimiter = ',')
    for x, y, z, r in table:
        blobs.append((np.array([x, y, z]), r))
    return blobs

def random_point_on_sphere(r):
    '''
        Generate random point on the sphere
        Parameters:
            r (float): radius of the sphere
        
        Returns:
            np.array: random point on the sphere
    '''
    u = np.random.rand()
    v = np.random.rand()
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def solid_sphere_w_blobs(blobs = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2*np.pi, 1000)
    v = np.linspace(0, np.pi, 500)

    # create the sphere surface
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones(np.size(u)), np.cos(v))

    if blobs is None:
        blobs = generate_blobs(10, 0.3, 0.2)

    heatmap = X.copy()
    for i in range( len( X ) ):
        for j in range( len( X[0] ) ):
            x = X[i, j]
            y = Y[i, j]
            z = Z[i, j]
            heatmap[i, j] = is_in_blob(np.array([x, y, z]), blobs )
    heatmap = heatmap / np.amax( heatmap )

    ax.plot_surface( X, Y, Z, cstride=1, rstride=1, facecolors=cm.cool(heatmap), alpha=0.15)
    plt.savefig('sphere3.png')
    # plt.show() 


def scatter_sphere_w_blobs(blobs=None):
    # Create data for the sphere
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x = np.outer(np.cos(u), np.sin(v))
    # y = np.outer(np.sin(u), np.sin(v))
    # z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x, y, z, color='b', alpha=1, zorder=1)

    if blobs is None:
        blobs = generate_blobs(10, 0.5, 0.1)
    table = np.loadtxt(path, delimiter = ',')

    for i, *x in table:
        i = int(i) 
        vec = np.array([1, 0, 0])
        R = np.array(x).reshape(3, 3)
        vec = R @ vec
        x, y, z = vec
        if is_in_blob(vec, blobs):
            ax.scatter(x, y, z, c='r', s=10, zorder=4)
        else:
            ax.scatter(x, y, z, c='g', s=10, zorder=4)

    plt.savefig('sphere2.png')
    # plt.show()


def sphere_by_set(path, filt, col):
    table = np.loadtxt(path, delimiter = ',')
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    # r = 1
    # pi = np.pi
    # cos = np.cos
    # sin = np.sin
    # phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    # x = r*sin(phi)*cos(theta)
    # y = r*sin(phi)*sin(theta)
    # z = r*cos(phi)

    # # ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='gray', alpha=0.6, linewidth=0)

    for i, *x in table:
        i = int(i) 
        vec = np.array([1, 0, 0])
        R = np.array(x).reshape(3, 3)
        vec = R @ vec
        color = col[filt[i%10]]
        ax.scatter([0, vec[0]], [0, vec[1]], [0, vec[2]], c = color, s = 1, alpha = 1)
    

    plt.savefig('sphere.png')
    # plt.show()


if __name__ == '__main__':
    path = 'matice.csv'
    filt = {0: 'val', 5: 'test', 1: 'train', 2: 'train', 3: 'train', 4: 'train', 6: 'train', 7: 'train', 8: 'train', 9: 'train'}
    col = {'train': 'lightblue', 'val': 'green', 'test': 'red'}
    sphere_by_set(path, filt, col)
    blobs = generate_blobs(10, 0.3, 0.1)
    save_blobs(blobs, 'blobs.csv')
    scatter_sphere_w_blobs(blobs)
    solid_sphere_w_blobs(blobs)

    '''
    momentalne bloby funguju:
    - vygenerujem nahodny bod na guli ako stred plus nahodny polomer
    - zisitm pre kazdu maticu kam zrotuje 1,0,0 vektor
    - hladam ci vzdialenost stredu blobu a zrotovaneho vektoru na povrchu gule je mensia ako polomer

    da sa:
    - vygenerovat nahodnu rotacnu maticu ako stred blobu plus nahodny uhol ako polomer
    - potom hladam uhol medzi maticami a zistujem ci je mensi ako uhol blobu
    - ale ked to budem plotovat aj tak budem rotovat 1,0,0 vektor 
    '''
W