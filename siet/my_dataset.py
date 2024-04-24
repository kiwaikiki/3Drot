from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import torch    
import psutil
# from matplotlib import pyplot as plt
class Dataset(Dataset):
    def __init__(self, path_pics : str, path_csv : str, split = 'train', width = 400, height = 400):
        self.path_pics = path_pics
        self.path_csv = path_csv
        self.split = split
        self.width = width
        self.height = height
        self.entries = None
        self.index2pic_id = None
        self.pictures = None

        self.load_matrices(path_csv)
        # self.load_pictures()

        filt = {'train': lambda x: x % 5 != 0, 'val': lambda x:  x % 10 == 0 , 'test': lambda x: x % 10 == 5}
        # filt = (lambda x: x % 5 != 0) if self.split == 'train' else (lambda x: x % 5 == 0)
        self.entries = {ind : mat for i, (ind, mat) in enumerate(self.entries.items()) if filt[self.split](i)}
        
        self.index2pic_id = np.array(list(self.entries.keys()))

        print("Split: ", self.split)
        print("Size: ", len(self))

    def load_matrices(self, path):
        table = np.loadtxt(path, delimiter = ',')
        # self.pic_id2index = {entry[0]: i for i, entry in enumerate(table)}
        # self.index2pic_id = table[:, 0].astype(int) 

        
        def help2matrix(x):
            return x[1:].reshape(3, 3)
        # 

        self.entries = dict(zip(table[:, 0].astype(int), np.apply_along_axis(help2matrix, arr = table, axis = 1)))
        # print(self.entries)
        # for i in self.entries:
        #     print(np.linalg.det(i))
    
    def load_pictures(self):
        pictures = []
        for i in self.index2pic_id:
            pictures.append(self.load_rgb(i))
            # print('cpu memory:', psutil.Process(os.getpid()).memory_info().rss)
        self.pictures = pictures


    def load_rgb(self, id_pic : int):
        """
        Loads pointcloud for a given entry
        :param entry: entry from self.entries
        :return: pointcloud wit shape (3, height, width)
        """
        exr_path = os.path.join(self.path_pics, f'{id_pic}.png')
        rgb = cv2.imread(exr_path)
        if rgb is None:
            print(exr_path)
            raise ValueError("Image at path ", exr_path)
        rgb = cv2.resize(rgb, (self.width, self.height))
        rgb = np.transpose(rgb, [2, 0, 1])
        rgb = rgb / 256
        # TODO loadnut do pamate vsetky obrazky
        return rgb
    
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
    
    def __getitem__(self, index): #fixme
        pic = self.load_rgb(self.index2pic_id[index])
        pic = pic.astype(np.float32)
        # print('**')
        # print(index)
        # print(self.index2pic_id[index])
        # print(self.entries[self.index2pic_id[index]])

        return {'index': self.index2pic_id[index], 
                'transform': torch.from_numpy(self.entries[self.index2pic_id[index]]),
                'pic' : pic}

    # def __getitem__(self, index): #fixme
    #     pic = self.pictures[index]
    #     pic = pic.astype(np.float32)

    #     return {'index': index, 
    #             'bin_transform': torch.from_numpy(self.entries[index]), 
    #             'pic' : pic}


if __name__ == "__main__":
    pic_dir = '../blendre/output/'
    csv_dir = '../blendre/matice.csv'
    dataset = Dataset(pic_dir, csv_dir, split = 'val')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)


    for item in data_loader:
        # print(item['xyz'].size())
        xyz = item['pic'][0].cpu().detach().numpy()
        # print(np.mean(xyz))

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(xyz[0].ravel(), xyz[1].ravel(), xyz[2].ravel(), marker='o')

        # plt.show ()