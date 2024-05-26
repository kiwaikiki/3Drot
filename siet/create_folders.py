import os
reprs = [
        # 'GS',
        # 'Euler',
        # 'Euler_binned',
        # 'Quaternion',
        # 'Axis_Angle_3D',
        'Axis_Angle_4D',
        # 'Axis_Angle_binned',
        # 'Stereographic',
        # 'Matrix'
    ]
losses = [
            # 'angle_rotmat',
            'elements',
            # 'angle_vectors',
            # 'elements2',

              ]

datasets = ['cube_cool', 
            'cube_big_hole', 
            'cube_dotted', 
            'cube_colorful', 
            'cube_one_color']

directory = 'training_data'
if not os.path.exists(directory):
    os.makedirs(directory)
    if os.path.exists(directory):
        print('done ', directory)

for r in reprs:
    path = os.path.join(directory, 'checkpoints', r)
    if not os.path.exists(path):
        os.makedirs(path)
    for l in losses:
        path = os.path.join(directory, 'checkpoints', r, l)
        if not os.path.exists(path):
            os.makedirs(path)
            if os.path.exists(path):
                print('done ', path)
# create folders in results
    path = os.path.join(directory, 'results', r)
    if not os.path.exists(path):
        os.makedirs(path)
    for l in losses:
        path = os.path.join(directory, 'results', r, l)
        if not os.path.exists(path):
            os.makedirs(path)
            if os.path.exists(path):
                print('done ', path)

    # create folders in inferences
    path = os.path.join(directory, 'inferences')
    if not os.path.exists(path):
        os.makedirs(path)
    for l in losses:
        path = os.path.join(directory, 'inferences', r, l)
        if not os.path.exists(path):
            os.makedirs(path)
            if os.path.exists(path):
                print('done ', path)


# for d in datasets:
#     path = os.path.join(directory, d)
#     if not os.path.exists(path):
#         os.makedirs(path)
#         if os.path.exists(path):
#             print('done ', path)
    
#     # create folders in checkpoints
#     for r in reprs:
#         path = os.path.join(directory,d, 'checkpoints', r)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         for l in losses:
#             path = os.path.join(directory,d, 'checkpoints', r, l)
#             if not os.path.exists(path):
#                 os.makedirs(path)
#                 if os.path.exists(path):
#                     print('done ', path)
    
#         # create folders in results
#         path = os.path.join(directory,d, 'results', r)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         for l in losses:
#             path = os.path.join(directory,d, 'results', r, l)
#             if not os.path.exists(path):
#                 os.makedirs(path)
#                 if os.path.exists(path):
#                     print('done ', path)

#         # create folders in inferences
#         path = os.path.join(directory,d, 'inferences')
#         if not os.path.exists(path):
#             os.makedirs(path)
#         for l in losses:
#             path = os.path.join(directory,d, 'inferences', r, l)
#             if not os.path.exists(path):
#                 os.makedirs(path)
#                 if os.path.exists(path):
#                     print('done ', path)

