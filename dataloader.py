import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
import time
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def log(string):
    print(time.strftime('%H:%M:%S'), ">> ", string)

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
            lr_path, hr_path = pair.split(",")
            hr_path = hr_path[:-1]
            name = lr_path.split("\\")[-1][:-4]
            lr_file = os.path.join(self.root, lr_path)
            hr_file = os.path.join(self.root, hr_path)
            self.files.append({
                "lr": lr_file,
                "hr": hr_file,
                "name": name
            })
        self.data = []
        if self.to_RAM:
            for i, fileinfo in enumerate(self.files):
                name = fileinfo["name"]
                lr_img = Image.open(fileinfo["lr"])
                hr_img = Image.open(fileinfo["hr"])
                self.data.append({
                    "lr": lr_img,
                    "hr": hr_img,
                    "name": name
                })
            log("Finish loading all images to RAM...")
 
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
            name = self.data[idx]["name"]
            lr_img = self.data[idx]["lr"]
            hr_img = self.data[idx]["hr"]
 
        '''random crop the inputs'''
        if self.crop_size > 0:

            #select a random start-point for croping operation
            h_offset = random.randint(0, lr_img.size[1] - self.crop_size)
            w_offset = random.randint(0, lr_img.size[0] - self.crop_size)
            #crop the image and the label
            crop_box = (w_offset, h_offset, w_offset+self.crop_size, h_offset+self.crop_size)
            lr_crop = lr_img.crop(crop_box)
            hr_crop = hr_img.crop(crop_box)
 
 
        '''convert PIL Image to numpy array'''
        lr_crop = np.asarray(lr_crop, np.float32).transpose((2,0,1))
        hr_crop = np.asarray(hr_crop, np.float32).transpose((2,0,1))
        return lr_crop, hr_crop, name

def build_LOLDataset_list_txt(dst_dir):
    log(f"Buliding LOLDataset list text at {dst_dir}")
    lr_dir = os.path.join(dst_dir, 'low')
    hr_dir = os.path.join(dst_dir, 'high')
    img_lr_path = [os.path.join('low', name) for name in os.listdir(lr_dir)]
    img_hr_path = [os.path.join('high', name) for name in os.listdir(hr_dir)]
    list_path = os.path.join(dst_dir, 'pair_list.csv')
    with open(list_path, 'w') as f:
        for lr_path, hr_path in zip(img_lr_path, img_hr_path):
            f.write(f"{lr_path},{hr_path}\n")
    log(f"Finish... There are {len(img_lr_path)} pairs...")
    return list_path

if __name__ == '__main__':
    root_path_train = r'H:\datasets\Low-Light Dataset\KinD++\LOLdataset\our485'
    root_path_test = r'H:\datasets\Low-Light Dataset\KinD++\LOLdataset\eval15'
    list_path_train = build_LOLDataset_list_txt(root_path_train)
    list_path_test = build_LOLDataset_list_txt(root_path_test)
    Batch_size = 2
    log("Buliding LOL Dataset...")
    dst_train = LOLDataset(root_path_train, list_path_train, transform=None, crop_size=200, to_RAM=True)
    dst_test = LOLDataset(root_path_test, list_path_test, transform=None, crop_size=200, to_RAM=True)
    # But when we are training a model, the mean should have another value
    trainloader = DataLoader(dst_train, batch_size = Batch_size)
    testloader = DataLoader(dst_test, batch_size=1)
    plt.ion()
    for i, data in enumerate(trainloader):
        _, imgs, name = data
        img = torchvision.utils.make_grid(imgs).numpy()
        img = img.astype(np.uint8) #change the dtype from float32 to uint8, because the plt.imshow() need the uint8
        img = np.transpose(img, (1, 2, 0))#transpose the Channels*H*W to  H*W*Channels
        #img = img[:, :, ::-1]
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
