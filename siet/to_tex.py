import numpy as np
import pandas as pd
import os

def table(cube, reprs, losses):
    text = '''
\\begin{table}[htbp]
\centering
\\begin{tabular}{||l c c c c c c||}
\\toprule
& \multicolumn{2}{c}{\\texttt{angle\_rotmat}} & \multicolumn{2}{c}{\\texttt{elements}} & \multicolumn{2}{c||}{\\texttt{angle\_vectors}}  \\\\
\midrule
    & Mean & Median & Mean & Median & Mean & Median \\\\
 '''
    for r in reprs:
        r_text = r.replace('_', '\_')    
        text += '\\texttt{' + r_text + '}'
        for l in losses:
            path = f'siet/training_data/{cube}/results/{r}/{l}/csv_results.csv'
            if not os.path.exists(path):
                text += '& - & -'
                continue
        
            table = pd.read_csv(path)
            index = table.last_valid_index()
            if index is None:
                text += '& - & -'
                continue
            mean = table.loc[index, 'mean']
            median = table.loc[index, 'median']
            text += f' & {mean:.2f} & {median:.2f}'
        text += ' \\\\ \n'
    
    text += '\\bottomrule \n \end{tabular}\n'
    cube_text = cube.replace('_', '\_') 
    text += f'\caption{{Results for {cube_text}}}'
    text += '\n\end{table}'

    return text

reprs = [
    'GS',
    'Euler',
    'Euler_binned',
    'Quaternion',
    'Axis_Angle_3D',
    'Axis_Angle_4D',
    'Axis_Angle_binned',
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

for dset in datasets:
    print(table(dset, reprs, losses))
    print('\n')