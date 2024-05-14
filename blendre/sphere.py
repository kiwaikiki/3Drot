from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.patches as mpatches

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
# def angle_between_rotations(R1, R2):
#     R = np.dot(R1, np.transpose(R2))
#     trace = np.trace(R)
#     trace = max(-1.0, min(trace, 3.0))
#     cos_angle = (trace - 1) / 2.0
#     angle = np.arccos(cos_angle)
#     return np.rad2deg(angle)


def rotation_angle(R):
    '''
    Calculate the rotation angle of a single tranformation by a 3x3 rotation matrix R
    '''
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def geodesic_distance(p1, p2, r=1):
    '''
        Calculate geodesic distance between two points on the sphere
        Parameters:
            p1 (np.array): first point
            p2 (np.array): second point
        
        Returns:
            float: geodesic distance between the points
    '''
    return r * np.arccos(np.clip(np.dot(p1, p2), -1, 1))

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

def evenly_spaced_rotation_matrices(n):
    spiral_points = golden_spiral(n, 1)
    matrices = []
    for point, r in spiral_points:
        matrices.append(rotation_matrix_from_vectors(np.array([1, 0, 0]), point))
    return matrices

def save_matrices(matrices, path):
    with open(path, 'w') as f:
        for R in matrices:
            f.write(','.join([str(x) for x in R.flatten()]) + '\n')

def load_matrices(path):
    matrices = []
    table = np.loadtxt(path, delimiter = ',')
    for x in table:
        matrices.append(np.array(x).reshape(3, 3))
    return matrices

def golden_spiral(n, radius):
    points = []
    inc = np.pi * (3 - np.sqrt(5))
    offset = 2 / n
    for i in range(n):
        y = i * offset - 1 + (offset / 2)
        r = np.sqrt(1 - y * y)
        phi = i * inc
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append(((x, y, z), radius))
    return points

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

    # ax.plot_surface( X, Y, Z, cstride=1, rstride=1, color=plt.cm.cool(0.5), alpha=0.9)
    ax.plot_surface( X, Y, Z, cstride=1, rstride=1, facecolors=cm.cool(heatmap), alpha=0.15)

    pink_patch = mpatches.Patch(color=plt.cm.cool(0.0), label='train + validation set')
    cyan_patch = mpatches.Patch(color=plt.cm.cool(1.0), label='test set')
    # purple_patch = mpatches.Patch(color=plt.cm.cool(0.5), label='all sets')

    plt.legend(handles=[pink_patch, cyan_patch], bbox_to_anchor=(0.95, 0.92), fontsize=20)
    
    plt.savefig('sphere3.png')
    plt.show() 

def solid_sphere_w_matrices(matrices, angle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2*np.pi, 1000)
    v = np.linspace(0, np.pi, 500)

    # create the sphere surface
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones(np.size(u)), np.cos(v))

    heatmap = X.copy()
    for i in range( len( X ) ):
        for j in range( len( X[0] ) ):
            x = X[i, j]
            y = Y[i, j]
            z = Z[i, j]
            matrix_to_point = rotation_matrix_from_vectors(np.array([1, 0, 0]), np.array([x, y, z]))
            heatmap[i, j] = is_in_blob_matrix(matrix_to_point, matrices, angle)
    # heatmap = heatmap / np.amax( heatmap )

    ax.plot_surface( X, Y, Z, cstride=1, rstride=1, facecolors=cm.cool(heatmap), alpha=0.15)

    pink_patch = mpatches.Patch(color=plt.cm.cool(0.0), label='train + validation set')
    cyan_patch = mpatches.Patch(color=plt.cm.cool(1.0), label='test set')
    # purple_patch = mpatches.Patch(color=plt.cm.cool(0.5), label='all sets')

    plt.legend(handles=[pink_patch, cyan_patch], bbox_to_anchor=(0.95, 0.92), fontsize=20)
    
    plt.savefig('sphere3.png')
    plt.show()


def is_in_blob_matrix(matrix, matrices, angle):
    for mat in matrices:
        rot_ang = rotation_angle(mat.T @ matrix)
        # print(rot_ang)
        if rot_ang < angle:
            vec1 = np.array([1, 0, 0])
            vec2 = np.array([1, 0, 0])
            vec1 = mat @ vec1
            vec2 = matrix @ vec2
            # print(geodesic_distance(vec1, vec2))
            return True
    return False

def rotation_matrix_from_vectors(v1, v2):
    '''
        Calculate the rotation matrix that rotates v1 to v2
        Parameters:
            v1 (np.array): vector to rotate
            v2 (np.array): target vector
        
        Returns:
            np.array: rotation matrix
    '''
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew * (1 - c) / (s ** 2)
        
def scatter_sphere_w_matrices(path, matrices=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if matrices is None:
        matrices = evenly_spaced_rotation_matrices(10)

    table = np.loadtxt(f'{path}/train/matice.csv', delimiter = ',')
    for i, *x in table:
        R = np.array(x).reshape(3, 3)
        vec = np.array([1, 0, 0])
        vec = R @ vec
        x, y, z = vec
        if is_in_blob_matrix(R, matrices, 40):
            ax.scatter(x, y, z, c='r')       
        else:
            ax.scatter(x, y, z, c='g')

    plt.savefig('sphere2.png')
    plt.show()

def scatter_sphere_w_blobs(blobs=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if blobs is None:
        blobs = generate_blobs(10, 0.5, 0.1)
    table = np.loadtxt(f'{path}/train/matice.csv', delimiter = ',')

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

def sphere_by_set(path):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    
    def f(x):
        vec = np.array([1, 0, 0])
        R = np.array(x[1:]).reshape(3, 3)
        return R @ vec 
    
    ax.scatter(0.66933941, 0.66848188, 0.3338211, c='black', s=50, zorder=5)
    
    table_test = np.loadtxt(f'{path}/test/matice.csv', delimiter = ',')
    matrices = np.apply_along_axis(f, 1, table_test)
    ax.scatter(matrices[:, 0], matrices[:, 1], matrices[:, 2], c=plt.cm.cool(1.0), label='test set', alpha=1, zorder=2)

    table_val = np.loadtxt(f'{path}/val/matice.csv', delimiter = ',')
    matrices = np.apply_along_axis(f, 1, table_val)
    ax.scatter(matrices[:, 0], matrices[:, 1], matrices[:, 2], c=plt.cm.cool(0.5), label='validation set', alpha=1, zorder=3)

    table_train = np.loadtxt(f'{path}/train/matice.csv', delimiter = ',')
    matrices = np.apply_along_axis(f, 1, table_train)
    ax.scatter(matrices[:, 0], matrices[:, 1], matrices[:, 2], c="lightblue", label='train set', alpha=0.2, zorder=4)

    lightblue_patch = mpatches.Patch(color='lightblue', label='train set')
    pink_patch = mpatches.Patch(color=plt.cm.cool(1.0), label='test set')
    purple_patch = mpatches.Patch(color=plt.cm.cool(0.5), label='validation set')

   
    plt.legend(handles=[pink_patch, purple_patch, lightblue_patch], bbox_to_anchor=(0.95, 0.92), fontsize=20)
    plt.savefig('sphere.png', bbox_inches='tight')
    plt.show()

def check_matrices(matrices):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')

    for mat in matrices:
        vec = np.array([1, 0, 0])
        vec = mat @ vec
        x, y, z = vec
        ax.scatter(x, y, z, c='r', s=10, zorder=4)    

    plt.show()

if __name__ == '__main__':
    path = 'cube_quad'
    sphere_by_set(path)
    # matrices = evenly_spaced_rotation_matrices(20)
    # save_matrices(matrices, 'matrices.csv')
    matrices = [rotation_matrix_from_vectors(np.array([1, 0, 0]), np.array([0, -1, 0]))]
    # matrices = load_matrices('matrices.csv')
    # check_matrices(matrices)
    # blobs = load_blobs('blobs.csv')
    # blobs = [(np.array([1, 0, 0]), 1)]
    # blobs = golden_spiral(20, 0.2)
    # save_blobs(blobs, 'blobs.csv')
    # scatter_sphere_w_blobs(blobs)
    # solid_sphere_w_blobs(blo////bs)
    # scatter_sphere_w_matrices(path, matrices)
    solid_sphere_w_matrices(matrices,50)


    # roatate around x axis 90
    # R_1 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # # rotate around y axis 90
    # R_2 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    # print(rotation_angle(R_1))
    # print(rotation_angle(R_2))
    # print(rotation_angle(R_1.T @ R_2))
    
    '''
    momentalne bloby funguju:
    - vygenerujem nahodny bod na guli ako stred plus nahodny polomer
    - zisitm pre kazdu maticu kam zrotuje 1,0,0 vektor
    - hladam ci vzdialenost stredu blobu a zrotovaneho vektoru na povrchu gule je mensia ako polomer

    da sa:
    - vygenerovat nahodnu rotacnu maticu ako stred blobu plus nahodny uhol ako polomer
    - potom hladam uhol medzi maticami a zistujem ci je mensi ako uhol blobu
    - ale ked to budem plotovat aj tak budem rotovat 1,0,0 vektor 

    ....well actually mozno neda
    '''
W