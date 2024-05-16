import numpy as np

def rotation_angle(R):
    '''
    Calculate the rotation angle of a single tranformation by a 3x3 rotation matrix R
    '''
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

def help2matrix(x):
    return x[1:].reshape(3, 3)


def closest(test_matrices, train_matrices, path_output):
    all_angles = []
    with open(path_output, 'w') as f:
        for i, test_matrix in enumerate(test_matrices):
            min_angle = (1000, -1)
            for j, train_matrix in enumerate(train_matrices):
                angle = rotation_angle(test_matrix.T @ train_matrix)
                if angle < min_angle[0]:
                    min_angle = (angle, j)
            
            all_angles.append(min_angle[0])
            f.write(f'{i},{min_angle[1]},{min_angle[0]}\n')
    return all_angles


datasets = ['cube_cool', 'cube_big_hole', 'cube_dotted', 'cube_colorful', 'cube_one_color']
for dataset in datasets:    
    path_test = f'datasets/{dataset}/test/matice.csv'
    path_train = f'datasets/{dataset}/train/matice.csv'
    table_test = np.loadtxt(path_test, delimiter=',')
    test_matrices = np.apply_along_axis(help2matrix, 1, table_test)
    table_train = np.loadtxt(path_train, delimiter=',')
    train_matrices = np.apply_along_axis(help2matrix, 1, table_train)

    path_output = f'datasets/{dataset}/closest.csv'
    all_angles = closest(test_matrices, train_matrices, path_output)
    # print(all_angles)
    print(dataset, 'mean: ', np.mean(all_angles))
    print(dataset, 'meadian: ', np.median(all_angles))

'''
cube_cool mean:  6.439367412262159
cube_cool meadian:  6.301196977254207
cube_big_hole mean:  18.980081086698906
cube_big_hole meadian:  17.50317113542893
cube_dotted mean:  9.878890932659946
cube_dotted meadian:  9.850634916720388
cube_colorful mean:  6.439367412262159
cube_colorful meadian:  6.301196977254207
cube_one_color mean:  6.439367412262159
cube_one_color meadian:  6.301196977254207
'''

