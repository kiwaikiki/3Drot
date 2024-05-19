import pandas as pd
import os

reprs = [
    'GS',
    'Euler',
    'Euler_binned',
    'Quaternion',
    'Axis_Angle_3D',
    'Axis_Angle_4D',
    # 'Axis_Angle_binned',
    # 'Stereographic',
    # 'Matrix'
]
losses = [
    'angle_rotmat',
    'elements',
    'angle_vectors'
    ]

datasets = [
        'cube_cool', 
        'cube_big_hole', 
        'cube_dotted', 
        'cube_colorful', 
        'cube_one_color'
        ]

# create table of all results merged
table = pd.DataFrame()
for dset in datasets:
    for repre in reprs:
        for loss_type in losses:
            path = f'siet/training_data/{dset}/results/{repre}/{loss_type}/csv_results.csv'
            if os.path.exists(path):
                tab = pd.read_csv(path)
                table = pd.concat([table, tab], ignore_index=True)
table_only_100 = table[table['epoch'] == 100]
table_only_100.to_csv('siet/training_data/all_results.csv', index=False)
