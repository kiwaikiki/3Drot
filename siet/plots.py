from matplotlib import pyplot as plt
import numpy as np

def plot_losses(train, val, title):
    with open(train, 'r') as f:
        train_lines = f.readlines()
        train_lines = [float(line.strip()) for line in train_lines]
    with open(val, 'r') as f:
        val_lines = f.readlines()
        val_lines = [float(line.strip()) for line in val_lines]
        plt.plot(train_lines, label='train')
        plt.plot(val_lines, label='val')
        plt.title(title)
        plt.legend()
        plt.savefig(f'{title}.png')
        plt.show()
        
    
if __name__ == "__main__":
    plot_losses('results/GS_train_err.out', 'results/GS_val_err.out', 'GS_train_val_err')