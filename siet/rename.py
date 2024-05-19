# rename all the directories that end with 'GS/angle' to 'GS/angle_rotmat'

# Path: bc/siet/rename.py
import os
reprs = [
        'GS',
        'Euler',
        'Euler_binned',
        'Quaternion',
        'Axis_Angle_3D',
        'Axis_Angle_4D',
        'Axis_Angle_binned',
        'Stereographic',
        'Matrix'
    ]
losses = ['angle', 'elements']
datasets = ['cube_cool',
            'cube_big_hole',
            'cube_dotted',
            'cube_colorful',
            'cube_one_color']

changes = [(('GS', 'angle'), ('GS', 'angle_rotmat')), 
           (('GS', 'elements'), ('GS', 'angle_vectors')),
           (('Euler', 'angle'), ('Euler', 'angle_rotmat')),
              (('Euler_binned', 'angle'), ('Euler_binned', 'angle_rotmat')),
              (('Quaternion', 'angle'), ('Quaternion', 'angle_rotmat')),
              (('Axis_Angle_3D', 'angle'), ('Axis_Angle_3D', 'angle_rotmat')),
              (('Axis_Angle_4D', 'angle'), ('Axis_Angle_4D', 'angle_rotmat')),
              (('Axis_Angle_binned', 'angle'), ('Axis_Angle_binned', 'angle_rotmat')),
              (('Stereographic', 'angle'), ('Stereographic', 'angle_rotmat')),
              (('Matrix', 'angle'), ('Matrix', 'angle_rotmat')),
           ]

for d in datasets:
    for r in reprs:
        for l in losses:
            for c in changes:
                if (r, l) == c[0]:
                    old_path = os.path.join('training_data', d, 'results', r, l)
                    new_path = os.path.join('training_data', d, 'results', c[1][0], c[1][1])
                    os.rename(old_path, new_path)
                    print(f'{old_path} -> {new_path}')
                if (r, l) == c[0]:
                    old_path = os.path.join('training_data', d, 'checkpoints', r, l)
                    new_path = os.path.join('training_data', d, 'checkpoints', c[1][0], c[1][1])
                    os.rename(old_path, new_path)
                    print(f'{old_path} -> {new_path}')
                if (r, l) == c[0]:
                    old_path = os.path.join('training_data', d, 'inferences', r, l)
                    new_path = os.path.join('training_data', d, 'inferences', c[1][0], c[1][1])
                    os.rename(old_path, new_path)
                    print(f'{old_path} -> {new_path}')

# #  in trainng data remove all the directories that end with '/elementsangle_vectors' 
# import os
# reprs = [
#         'GS',
#         'Euler',
#         'Euler_binned',
#         'Quaternion',
#         'Axis_Angle_3D',
#         'Axis_Angle_4D',
#         'Axis_Angle_binned',
#         'Stereographic',
#         'Matrix'
#     ]
# losses = ['angle', 'elements', 'angle_vectors']
# datasets = ['cube_cool',
#             'cube_big_hole',
#             'cube_dotted',
#             'cube_colorful',
#             'cube_one_color']

# for d in datasets:
#     for r in reprs:
#         l = 'elementsangle_vectors'
#         path = os.path.join('training_data', d, 'results', r, l)
#         if os.path.exists(path):
#             os.rmdir(path)
#             print(f'{path} removed')
#         path = os.path.join('training_data', d, 'checkpoints', r, l)
#         if os.path.exists(path):
#             os.rmdir(path)
#             print(f'{path} removed')
#         path = os.path.join('training_data', d, 'inferences', r, l)
#         if os.path.exists(path):
#             os.rmdir(path)
#             print(f'{path} removed')