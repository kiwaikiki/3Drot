from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import torch    
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
class Dataset(Dataset):
    def __init__(self, path_pics : str, path_csv : str, split = 'train', width = 256, height = 256):
        self.path_pics = path_pics
        self.path_csv = path_csv
        self.split = split
        self.width = width
        self.height = height
        self.entries = None
        self.pictures = None

        self.load_matrices()
        # self.load_pictures()

        print("Split: ", self.split)
        print("Size: ", len(self))

    def load_matrices(self):
        path = os.path.join(self.path_pics, self.split, self.path_csv)
        table = np.loadtxt(path, delimiter = ',')
        def help2matrix(x):
            return x[1:].reshape(3, 3)
        self.entries = np.apply_along_axis(help2matrix, arr = table, axis = 1)
        
    def load_pictures(self):
        pictures = []
        for i in range(len(self.entries)):
            pictures.append(self.load_rgb(i))
        self.pictures = pictures

    def load_rgb(self, id_pic : int):
        """
        Loads pointcloud for a given entry
        :param entry: entry from self.entries
        :return: pointcloud wit shape (3, height, width)
        """
        exr_path = os.path.join(self.path_pics, self.split, f'{id_pic:04d}.png')
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

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, index): #fixme
        pic = self.load_rgb(index)
        pic = pic.astype(np.float32)
        
        return {'index': index, 
                'transform': torch.from_numpy(self.entries[index]),
                'pic' : pic
                # 'pic' : self.pictures[index].astype(np.float32)
                }

def display_picture(pic):
    pic = np.transpose(pic, [1, 2, 0])
    plt.imshow(pic[:,:,::-1])
    plt.savefig('pic.png')
    plt.show()


if __name__ == "__main__":
    pic_dir = '../cube_quad2'
    csv_dir = 'matice.csv'
    dataset = Dataset(pic_dir, csv_dir, split = 'val')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for item in data_loader:
        print(item['index'][0])
        print(item['transform'][0])
        xyz = item['pic'][0].cpu().detach().numpy()
        display_picture(xyz)
        break
       