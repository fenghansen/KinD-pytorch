import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
 
class LOLDataset(Dataset):
    def __init__(self, root, list_path, transform, crop_size=256, to_RAM=False):
        super(LOLDataset,self).__init__()
        self.to_RAM = to_RAM
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        with open(list_path) as f:
            self.pairs = f.readlines()
        self.files = []
        for pair in self.pairs:
            lr_path, hr_path = pair.split(" ")
            name = lr_path.split("\\")[-1][:-4]
            lr_file = os.path.join(self.root, lr_path)
            hr_file = os.path.join(self.root, hr_path)
            self.files.append({
                "lr": lr_file,
                "hr": hr_file,
                "name": name
            })
 
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, idx):
        datafiles = self.files[idx]
 
        '''load the datas'''
        if not self.to_RAM:
            name = datafiles["name"]
            lr_img = Image.open(datafiles["lr"])
            hr_img = Image.open(datafiles["hr"])
        else:
            pass
 
        '''random crop the inputs'''
        if self.crop_size > 0:

            #select a random start-point for croping operation
            h_offset = random.randint(0, lr_img.size[0] - self.crop_size)
            w_offset = random.randint(0, lr_img.size[1] - self.crop_size)
            #crop the image and the label
            crop_box = (w_offset, h_offset, w_offset+self.crop_size, h_offset+self.crop_size)
            lr_crop = lr_img.crop(crop_box)
            hr_crop = hr_img.crop(crop_box)
 
 
        '''convert PIL Image to numpy array'''
        lr_crop = np.asarray(lr_crop, np.float16).transpose((2,0,1))
        hr_crop = np.asarray(hr_crop, np.float16).transpose((2,0,1))
        return lr_crop, hr_crop, name
 
if __name__ == '__main__':
    DATA_DIRECTORY = '/home/teeyo/STA/Data/voc_aug/'
    DATA_LIST_PATH = '../dataset/list/val.txt'
    Batch_size = 4
    dst = VOCDataSet(DATA_DIRECTORY,DATA_LIST_PATH, mean=(0,0,0))
    # just for test,  so the mean is (0,0,0) to show the original images.
    # But when we are training a model, the mean should have another value
    trainloader = DataLoader(dst, batch_size = Batch_size)
    plt.ion()
    for i, data in enumerate(trainloader):
        imgs, labels,_,_= data
        if i%1 == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = img.astype(np.uint8) #change the dtype from float32 to uint8, because the plt.imshow() need the uint8
            img = np.transpose(img, (1, 2, 0))#transpose the Channels*H*W to  H*W*Channels
            #img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            plt.pause(0.5)
