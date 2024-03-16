from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
import os
import torch    
# from matplotlib import pyplot as plt
class Dataset(Dataset):
    def __init__(self, path_pics : str, path_csv : str, split = 'train', width = 400, height = 400):
        self.path_pics = path_pics
        self.path_csv = path_csv
        self.split = split
        self.width = width
        self.height = height
        self.entries = None
        self.load(path_csv)

        if self.split == 'train':
            self.entries = [entry for i, entry in enumerate(self.entries) if i % 5 != 0]
        elif self.split == 'val':
            self.entries = [entry for i, entry in enumerate(self.entries) if i % 5 == 0]

        print("Split: ", self.split)
        print("Size: ", len(self))
        

    def load(self, path):
        table = pd.read_csv(path)
        def help2matrix(x):
            return np.array([[x[1], x[2], x[3]],
                              [x[4], x[5], x[6]],
                              [x[7], x[8], x[9]]])
        self.entries = np.apply_along_axis(help2matrix, arr = table, axis = 1)

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
        xyz = cv2.resize(xyz, (self.width, self.height), cv2.INTER_NEAREST_EXACT)
        xyz = np.transpose(xyz, [2, 0, 1])
        # xyz = self.add_sur_2(xyz)
        return xyz
    
    def add_sur(self, xyz):
        h_sur, w_sur = np.indices((self.height, self.width))
        xyz = np.dstack((xyz, h_sur, w_sur))
        print(xyz.shape)
        return xyz
    
    def add_sur_2(self, xyz):
        h_sur, w_sur = np.indices((self.height, self.width))
        r = xyz[:,:,0]
        g = xyz[:,:,1]
        b = xyz[:,:,2]
        # xyz = np.array([r, g, b, h_sur, w_sur])
        xyz = np.array([r, g, b])
        return xyz

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, index):
        xyz = self.load_xyz(index)
        xyz = xyz.astype(np.float32)

        return {'id': index, 
                'bin_transform': torch.from_numpy(self.entries[index]), 
                'xyz' : xyz}


if __name__ == "__main__":
    pic_dir = '/home/viki/FMFI/bc/blendre/output/'
    csv_dir = '/home/viki/FMFI/bc/blendre/matice.csv'
    dataset = Dataset(pic_dir, csv_dir)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)


    for item in data_loader:
        # print(item['xyz'].size())
        xyz = item['xyz'][0].cpu().detach().numpy()
        print(np.mean(xyz))

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(xyz[0].ravel(), xyz[1].ravel(), xyz[2].ravel(), marker='o')

        # plt.show ()