from matplotlib import pyplot as plt
import numpy as np
import os

def plot_losses(path, train, val, test, title):
    with open(f'{path}/{train}', 'r') as f:
        train_lines = f.readlines()
        train_lines = [float(line.strip()) for line in train_lines]

    with open(f'{path}/{val}', 'r') as f:
        val_lines = f.readlines()
        val_lines = [float(line.strip()) for line in val_lines]
        plt.plot(train_lines, label='train')
        plt.plot(val_lines, label='val')
        
    with open(f'{path}/{test}', 'r') as f:
        test_lines = np.loadtxt(f, delimiter=',')
        plt.plot(test_lines.T[0], test_lines.T[1], label='test')
    
    plt.title(title)
    plt.legend()
    plt.savefig(f'{path}/{title}.png')
    plt.show()

    
if __name__ == "__main__":
    dataset = 'cool_cube'
    repre = 'Euler'
    lossf = 'angle'
    path = os.path.join('siet', 'training_data', dataset, 'results', repre, lossf)
    plot_losses(path, 'train_err.out', 'val_err.out', 'test_err_by_epochs.csv', f'{repre} {lossf}')