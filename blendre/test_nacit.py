import os
import cv2
import bpy
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from blend import PictureGenerator as pg 
import math  # noqa: E402
        
def load_rgb(id_pic : int, path):
        """
        Loads pointcloud for a given entry
        :param entry: entry from self.entries
        :return: pointcloud wit shape (3, height, width)
        """
        exr_path = os.path.join(path, f'{id_pic:04d}.png')
        rgb = cv2.imread(exr_path)
        if rgb is None:
            print(exr_path)
            raise ValueError("Image at path ", exr_path)
        rgb = cv2.resize(rgb, (256, 256))
        rgb = np.transpose(rgb, [2, 0, 1])
        rgb = rgb / 256
        return rgb


def load_matrices(path):
        table = np.loadtxt(path, delimiter = ',')
        
        def help2matrix(x):
            return np.array([[x[1], x[2], x[3]],
                              [x[4], x[5], x[6]],
                              [x[7], x[8], x[9]]])
        
        return np.apply_along_axis(help2matrix, arr = table, axis = 1)

def display_picture(pic):
    pic = np.transpose(pic, [1, 2, 0])
    plt.imshow(pic[:,:,::-1])
    plt.savefig('test_nacitane.png')


def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


if __name__ == "__main__":
    index = 1
    
    print(index)
    path = 'cube_dotted/train'
    picture = load_rgb(index, path)
    display_picture(picture)
    
    matrix = load_matrices('cube_dotted/train/matice.csv')
    matica = matrix[index]
    print(matica)

    euler = rotationMatrixToEulerAngles(matica)
    print(euler)
    
    obj = pg('modely/kocky_texture.fbx', (400, 400), 'textury/cool_voronoi.png')
    previous_mode = obj.obj.rotation_mode 
    obj.obj.rotation_mode = "XYZ"
    obj.obj.rotation_euler = euler
    obj.obj.rotation_mode  = previous_mode
    bpy.context.scene.render.filepath = 'test_z_uhlov.png'
    bpy.ops.render.render(write_still=True)


