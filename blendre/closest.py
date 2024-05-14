import numpy as np
from sphere import rotation_angle


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


dataset = 'cube_quad'
path_test = f'{dataset}/test/matice.csv'
path_train = f'{dataset}/train/matice.csv'
table_test = np.loadtxt(path_test, delimiter=',')
test_matrices = np.apply_along_axis(help2matrix, 1, table_test)
table_train = np.loadtxt(path_train, delimiter=',')
train_matrices = np.apply_along_axis(help2matrix, 1, table_train)

path_output = f'{dataset}/closest.csv'
all_angles = closest(test_matrices, train_matrices, path_output)
print(all_angles)
print(np.mean(all_angles))
