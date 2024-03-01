import bpy
import os
from bpy import context, data, ops
import numpy as np
from io import StringIO   # StringIO behaves like a file object
import time

PROJECT_DIR = '/home/viki/FMFI/bc/blendre/'
os.chdir(PROJECT_DIR)

def render_object_from_angles(obj, output_dir, matrix_file):
    with open(matrix_file, 'w') as f:
        for id, angle in enumerate(generate_uniform_angles(100)):
            start_time = time.time()    
            obj.rotation_euler = angle
            matrix = matrix2string(obj.matrix_world) 
            print(f'{id}\n{matrix}', file = f)
            bpy.context.scene.render.filepath = os.path.join(output_dir, f"{id}.png")
            bpy.ops.render.render(write_still=True)
            print(f'{id} took {time.time() - start_time} seconds')

def matrix2string(matrix):
    return '\n'.join([' '.join(map(str, row[:-1])) for row in matrix[:-1]])   

def cleanup_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action='DESELECT')

def add_texture(ob):
    mat = bpy.data.materials.new(name="New_Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load('gray_rocks_4k.blend/textures/gray_rocks_nor_gl_4k.exr')
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)

def set_scene():
    bpy.ops.object.light_add(type='POINT', location=(5, -10, 5))
    light = bpy.context.object
    light.data.energy = 5000
    bpy.ops.object.camera_add(location=(0, -15, 0))
    bpy.data.objects['Camera'].rotation_euler = (1.5708, 0, 0)  # Rotate camera to point downwards
    bpy.context.scene.camera = bpy.context.object

def import_object(file_path):
    bpy.ops.import_scene.obj(filepath=file_path)
    obj = bpy.context.selected_objects[0]
    return obj

def set_rendering_settings(x, y):
    bpy.context.scene.render.resolution_x = x
    bpy.context.scene.render.resolution_y = y

def generate_uniform_angles(n):
    for _ in range(n):
        yield (np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi))

def generate():
    cleanup_scene()
    set_scene()
    obj = import_object("modely/kocky_vstrede.obj")
    add_texture(obj)
    set_rendering_settings(400, 400)

    render_object_from_angles(obj, "output", 'matice.txt')

if __name__ == "__main__":
    generate()

# check
    with open('matice.txt', 'r') as f:
        l = ''.join(f.readlines()[1:4])
        c = StringIO(l)
        M = np.loadtxt(c)
        print(np.linalg.det(M))
        print(M@M.T)



    
'''
rotacna matica z blendru
optimalizacia
resnet 18
pytorch nacitat a spustit

cv4 
custom torch module
'''