from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import os

class Dataset(Dataset):
    def __init__(self, path_pics : str, path_csv : str):
        self.path_pics = path_pics
        self.path_csv = path_csv
        self.width = 400
        self.height = 400
        self.matrices = None

        self.load(path_csv)

    def load(self, path):
        table = pd.read_csv(path)
        def help_func(x):
            return np.array([[x[1], x[2], x[3]],
                              [x[4], x[5], x[6]],
                              [x[7], x[8], x[9]]])
        self.matrices = np.apply_along_axis(help_func, arr = table, axis = 1)


    def load_xyz(self, id_pic : int):
        """
        Loads pointcloud for a given entry
        :param entry: entry from self.entries
        :return: pointcloud wit shape (3, height, width)
        """
        exr_path = os.path.join(self.path_pics, f'{id_pic}.png')
        xyz = cv2.imread(exr_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if xyz is None:
            print(exr_path)
            raise ValueError("Image at path ", exr_path)
        xyz = self.add_sur(xyz)
        return xyz
    
    def add_sur(self, xyz):
        h_sur, w_sur = np.indices((self.height, self.width))
        xyz = np.dstack((xyz, h_sur))
        xyz = np.dstack((xyz, w_sur))
        return xyz

    

if __name__ == "__main__":
    pic_dir = '/home/viki/FMFI/bc/blendre/output/'
    csv_dir = '/home/viki/FMFI/bc/blendre/matice.csv'
    dataset = Dataset(pic_dir, csv_dir)
    dataset.load_xyz(5)