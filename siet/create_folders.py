import os
reprs = [
        'GS',
        'Euler',
        'Euler_binned',
        'Quaternion',
        'Axis_Angle',
        'Axis_Angle_binned',
        'Stereographic',
        'Matrix'
    ]
losses = ['angle', 'elements']

datasets = ['cool_cube', 
            'big_hole_cube', 
            'dotted_cube', 
            'colorful_cube', 
            'one_color_cube']

directory = 'training_data'
if not os.path.exists(directory):
    os.makedirs(directory)
    if os.path.exists(directory):
        print('done ', directory)

for d in datasets:
    path = os.path.join(directory, d)
    if not os.path.exists(path):
        os.makedirs(path)
        if os.path.exists(path):
            print('done ', path)
    
    # create folders in checkpoints
    for r in reprs:
        path = os.path.join(directory,d, 'checkpoints', r)
        if not os.path.exists(path):
            os.makedirs(path)
        for l in losses:
            path = os.path.join(directory,d, 'checkpoints', r, l)
            if not os.path.exists(path):
                os.makedirs(path)
                if os.path.exists(path):
                    print('done ', path)
    
        # create folders in results
        path = os.path.join(directory,d, 'results', r)
        if not os.path.exists(path):
            os.makedirs(path)
        for l in losses:
            path = os.path.join(directory,d, 'results', r, l)
            if not os.path.exists(path):
                os.makedirs(path)
                if os.path.exists(path):
                    print('done ', path)

        # create folders in inferences
        path = os.path.join(directory,d, 'inferences')
        if not os.path.exists(path):
            os.makedirs(path)
        for l in losses:
            path = os.path.join(directory,d, 'inferences', r, l)
            if not os.path.exists(path):
                os.makedirs(path)
                if os.path.exists(path):
                    print('done ', path)

