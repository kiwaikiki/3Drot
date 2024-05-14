from matplotlib import pyplot as plt
import numpy as np

def plot_losses(path, train, val, test, title):
    with open(train, 'r') as f:
        train_lines = f.readlines()
        train_lines = [float(line.strip()) for line in train_lines]

    with open(val, 'r') as f:
        val_lines = f.readlines()
        val_lines = [float(line.strip()) for line in val_lines]
        plt.plot(train_lines, label='train')
        plt.plot(val_lines, label='val')
        
    with open(test, 'r') as f:
        test_lines = np.loadtxt(f, delimiter=',')
        plt.plot(test_lines.T[0], test_lines.T[1], label='test')
    
    plt.title(title)
    plt.legend()
    plt.savefig(f'{path}{title}.png')
    plt.show()

    
if __name__ == "__main__":
    path = 'results/Quaternion/angle/'
    plot_losses(path, f'{path}train_err.out', f'{path}val_err.out', f'{path}test_err_by_epochs.csv', 'Euler elements')