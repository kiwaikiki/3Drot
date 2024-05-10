import bpy
import os
from bpy import context, data, ops
import numpy as np
from io import StringIO   # StringIO behaves like a file object
import time
import pandas as pd
import multiprocessing as mp
from sphere import geodesic_distance, load_blobs, is_in_blob

PROJECT_DIR = '/home/viki/FMFI/bc/blendre/'
os.chdir(PROJECT_DIR)

class PictureGenerator:
    def __init__(self, obj_file : str,  render_x_y : tuple, texture_file : str = None) -> None:
        self.cleanup_scene()
        self.set_scene()
        self.set_rendering_settings(*render_x_y)
        self.obj = self.import_object(obj_file)
        if texture_file:
            self.add_texture(texture_file)

    def cleanup_scene(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        bpy.ops.object.select_all(action='DESELECT')

    def set_scene(self):
        '''
        add light and camera
        '''
        bpy.ops.object.light_add(type='POINT', location=(5, -10, 5))
        light = bpy.context.object
        light.data.energy = 3000
        bpy.ops.object.camera_add(location=(0, -15, 0))
        bpy.data.objects['Camera'].rotation_euler = (1.5708, 0, 0)  # Rotate camera to point downwards
        bpy.context.scene.camera = bpy.context.object
        

    def render_object_from_angles(self, output_dir, matrix_file, number_of_angles=100):
        with open(f'{output_dir}/{matrix_file}', 'w') as f:
            start_time = time.time()  
            for id, angle in enumerate(self.generate_uniform_angles(number_of_angles)):
            # for id, angle in enumerate(self.generate_angles_quad_train(number_of_angles)):
            # for id, angle in enumerate(self.generate_angles_quad_test(number_of_angles)):
                previous_mode = self.obj.rotation_mode 
                self.obj.rotation_mode = "XYZ"
                self.obj.rotation_euler = angle
                self.obj.rotation_mode  = previous_mode

                if matrix_file.endswith('.csv'):
                    matrix = matrix2csv(self.obj.matrix_world)
                    print(f'{id-1},{matrix}', file = f)
                else:  
                    matrix = matrix2string(self.obj.matrix_world) 
                    print(f'{id-1}\n{matrix}', file = f)
                bpy.context.scene.render.filepath = os.path.join(output_dir, f"{id:04d}.png")
                bpy.ops.render.render(write_still=True)

            print(f'Elapsed time: {time.time() - start_time}')

    def render_blobs(self, output_dir, blob_file, matrix_file, number_of_angles=100):
        with open(f'{output_dir}/{matrix_file}', 'w') as f:
            start_time = time.time()  
            blobs = load_blobs(blob_file)
            generated = 0
            while generated < number_of_angles:
                angle = (np.random.uniform(0, 2*np.pi), 
                        np.random.uniform(0, 2*np.pi), 
                        np.random.uniform(0, 2*np.pi))
                previous_mode = self.obj.rotation_mode 
                self.obj.rotation_mode = "XYZ"
                self.obj.rotation_euler = angle
                self.obj.rotation_mode  = previous_mode

                if matrix_file.endswith('.csv'):
                    matrix = matrix2csv(self.obj.matrix_world)
                    print(f'{id-1},{matrix}', file = f)
                else:  
                    matrix = matrix2string(self.obj.matrix_world) 
                    print(f'{id-1}\n{matrix}', file = f)
                bpy.context.scene.render.filepath = os.path.join(output_dir, f"{id:04d}.png")
                bpy.ops.render.render(write_still=True)

            print(f'Elapsed time: {time.time() - start_time}')



    def add_texture(self, texture_file=None):
        '''
        object should already have uv map
        '''
        mat = bpy.data.materials.new(name="New_Mat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage.image = bpy.data.images.load(texture_file)
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        if self.obj.data.materials:
            self.obj.data.materials[0] = mat
        else:
            self.obj.data.materials.append(mat)

    def import_object(self, ile_path):
        print(bpy.ops.import_scene.fbx(filepath='modely/kocky_texture.fbx'))
        # bpy.ops.import_scene.obj(filepath=file_path)
        bpy.ops.object.select_all(action='SELECT')
        obj = bpy.context.selected_objects[-1] # 0 is camera, 1 is light
        return obj

    def set_rendering_settings(self, x, y):
        bpy.context.scene.render.resolution_x = x
        bpy.context.scene.render.resolution_y = y

    def generate_angles_quad_train(self, n):
        for _ in range(n):
            yield (np.random.uniform(0, 1.5*np.pi),
                np.random.uniform(0, 1.5*np.pi), 
                np.random.uniform(-np.pi, 0.5*np.pi))
            
    def generate_angles_quad_test(self, n):
        for _ in range(n):
            yield (np.random.uniform(0, 0.5*np.pi),
                np.random.uniform(0.5*np.pi, np.pi), 
                np.random.uniform(0, 0.5*np.pi))

    def generate_uniform_angles(self, n):
        for _ in range(n):
            yield (np.random.uniform(0, 2*np.pi), 
                np.random.uniform(0, 2*np.pi), 
                np.random.uniform(0, 2*np.pi))

    def get_matrix_from_csv(self, id, table):
        return np.array(table.loc[id]).reshape(3, 3)

    def paralel_rendering(self, n=100):
        with mp.Pool(10) as p:
            p.map(help_paralel, zip(range(n), [self.obj.copy() for _ in range(n)]))

# def is_in_blob_matrix()

def help_paralel(args):
    obj_id, obj = args
    obj.rotation_euler = (np.random.uniform(0, 2*np.pi), 
                        np.random.uniform(0, 2*np.pi), 
                        np.random.uniform(0, 2*np.pi))
    bpy.context.scene.render.filepath = os.path.join('output', f"{obj_id}.png")
    bpy.ops.render.render(write_still=True)
    with open('matice.csv', 'w') as f:
        matrix = PictureGenerator.matrix2csv(obj.matrix_world)
        print(f'{obj_id},{matrix}', file = f)
        

def matrix2string(matrix):
    '''
    returns string parsable by numpy. 
    discards last row and column(translation).
    '''
    return '\n'.join([' '.join(map(str, row[:-1])) for row in matrix[:-1]])   

def matrix2csv(matrix):
    '''
    returns string parsable by numpy. 
    discards last row and column(translation).
    order first row, second row
    '''
    return ','.join([','.join(map(str, row[:-1])) for row in matrix[:-1]])


def rotation_Matrix2angles(R):
    '''
    Convert a rotation matrix to angles
    '''
    x = np.arctan2(R[2][1], R[2][2])
    y = np.arctan2(-R[2][0], np.sqrt(R[2][1]**2 + R[2][2]**2))
    z = np.arctan2(R[1][0], R[0][0])
    return np.array([x, y, z])


if __name__ == "__main__":
    gen = PictureGenerator('modely/kocky_texture.fbx', (256, 256), 'textury/cool_voronoi.png')
    
    gen.render_object_from_angles('cube_quad/train', 'matice.csv', 500)
    # gen.render_object_from_angles('cube_quad/val', 'matice.csv', 100)
    # gen.render_object_from_angles('cube_quad/test', 'matice.csv', 100)  
#     # gen.paralel_rendering()
# # check
# #  
#     with open('matice2.csv', 'r') as f:
#         table = pd.read_csv(f, header=None, index_col=0)
#         M = PictureGenerator.get_matrix_from_csv(2, table)
#         print(M)
#         print(np.linalg.det(M))
#         print(M@M.T)
            



    
'''
rotacna matica z blendru
optimalizacia
resnet 18
pytorch nacitat a spustit

cv4 
custom torch module
'''