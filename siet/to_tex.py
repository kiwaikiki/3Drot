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
    'Euler',
    'Euler_binned',
    'Quaternion',
    'Axis_Angle_3D',
    'Axis_Angle_4D',
    'Axis_Angle_binned',
    'Stereographic',
    'GS',

    # 'Matrix'
]

losses = [
        'angle_rotmat',
        'elements2',
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

def other_table(cube, reprs, losses):
    text = '''
\\begin{table}[htbp]
\centering
\\begin{tabular}{||l c c c c c c||}
\\toprule
Reprezentácia & Chybová funkcia & Id & Medián & Priemer & AUC$5^\circ$ & AUC$10^\circ$ & AUC$20^\circ$ \\\\
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

def gpt_table(cube):

    cube_table = big_table[big_table['dataset'] == cube]
    selected_columns = cube_table[['id', 'represetnation', 'loss_function', 'mean', 'median', '5', '10', 'all']]
    selected_columns.columns = ['Id', 'Reprezentácia', 'Chybová funkcia', 'Priemer', 'Medián', 'AUC $5^\circ$', 'AUC $10^\circ$', 'AUC $20^\circ$']
    selected_columns['Reprezentácia'] = '\\texttt{' + selected_columns['Reprezentácia'].str.replace('_', '\_') + '}'
    selected_columns['Chybová funkcia'] = '\\texttt{' + selected_columns['Chybová funkcia'].str.replace('_', '\_') + '}'
    selected_columns['Priemer'] = selected_columns['Priemer'].map('{:.2f}'.format)
    selected_columns['Medián'] = selected_columns['Medián'].map('{:.2f}'.format)
    selected_columns['AUC $5^\circ$'] = selected_columns['AUC $5^\circ$'].map('{:.2f}'.format)
    selected_columns['AUC $10^\circ$'] = selected_columns['AUC $10^\circ$'].map('{:.2f}'.format)
    selected_columns['AUC $20^\circ$'] = selected_columns['AUC $20^\circ$'].map('{:.2f}'.format)


    latex_table = selected_columns.to_latex(index=False, column_format='||c c c c c c c c||', 
                                        header=True, escape=False)
    return latex_table



big_table = pd.read_csv(f'siet/training_data/all_results.csv')
big_table['represetnation'] = pd.Categorical(big_table['represetnation'], categories=reprs, ordered=True)
big_table['loss_function'] = pd.Categorical(big_table['loss_function'], categories=losses, ordered=True)
big_table = big_table.sort_values(by=['represetnation', 'loss_function'])

unique_combinations = big_table[['represetnation', 'loss_function']].drop_duplicates().reset_index(drop=True)
unique_combinations['id'] = range(1, len(unique_combinations) + 1)
big_table = big_table.merge(unique_combinations, on=['represetnation', 'loss_function'], how='left')


with open('tabulky.tex', 'w') as f:
    for dset in datasets:
        f.write(dset+'\n')
        f.write(gpt_table(dset))
        f.write('\n\n')
# for dset in datasets:
#     print(gpt_table(dset))
#     print('\n')