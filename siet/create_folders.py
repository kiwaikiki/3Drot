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

directory = 'bc/siet/'
# create folders in checkpoints
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

