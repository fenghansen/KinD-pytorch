import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataloader import *

Sobel = np.array([[-1,-2,-1],
                  [ 0, 0, 0],
                  [ 1, 2, 1]])
Robert = np.array([[0, 0],
                  [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)

def feature_map_hook(*args, path=None):
    feature_maps = []
    for feature in args:
        feature_maps.append(feature)
    feature_all = torch.cat(feature_maps, dim=1)
    fmap = feature_all.detach().cpu().numpy()[0]
    fmap = np.array(fmap)
    fshape = fmap.shape
    num = fshape[0]
    length = fshape[-1]
    sample(fmap, figure_size=(2, num//2), img_dim=length, path=path)
    return fmap

# 已测试本模块没有问题，作用为提取一阶导数算子滤波图（边缘图）
def gradient(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=1))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm

class Decom_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def reflectance_similarity(self, R_low, R_high):
        return torch.mean(torch.abs(R_low - R_high))
    
    def illumination_smoothness(self, I, L, name='_low', hook=-1):
        # L_transpose = L.permute(0, 2, 3, 1)
        # L_gray_transpose = 0.299*L[:,:,:,0] + 0.587*L[:,:,:,1] + 0.114*L[:,:,:,2]
        # L_gray = L.permute(0, 3, 1, 2)
        L_gray = 0.299*L[:,0,:,:] + 0.587*L[:,1,:,:] + 0.114*L[:,2,:,:]
        L_gray = L_gray.unsqueeze(dim=1)
        I_gradient_x = gradient(I, "x")
        L_gradient_x = gradient(L_gray, "x")
        epsilon = 0.01*torch.ones_like(L_gradient_x)
        Denominator_x = torch.max(L_gradient_x, epsilon)
        x_loss = torch.abs(torch.div(I_gradient_x, Denominator_x))
        I_gradient_y = gradient(I, "y")
        L_gradient_y = gradient(L_gray, "y")
        Denominator_y = torch.max(L_gradient_y, epsilon)
        y_loss = torch.abs(torch.div(I_gradient_y, Denominator_y))
        mut_loss = torch.mean(x_loss + y_loss)
        if hook > -1:
            feature_map_hook(I, L_gray, epsilon, I_gradient_x+I_gradient_y, Denominator_x+Denominator_y, 
                            x_loss+y_loss, path=f'./samples-features/ilux_smooth_{name}_epoch{hook}.png')
        return mut_loss
    
    def mutual_consistency(self, I_low, I_high, hook=-1):
        low_gradient_x = gradient(I_low, "x")
        high_gradient_x = gradient(I_high, "x")
        M_gradient_x = low_gradient_x + high_gradient_x
        x_loss = M_gradient_x * torch.exp(-10 * M_gradient_x)
        low_gradient_y = gradient(I_low, "y")
        high_gradient_y = gradient(I_high, "y")
        M_gradient_y = low_gradient_y + high_gradient_y
        y_loss = M_gradient_y * torch.exp(-10 * M_gradient_y)
        mutual_loss = torch.mean(x_loss + y_loss) 
        if hook > -1:
            feature_map_hook(I_low, I_high, low_gradient_x+low_gradient_y, high_gradient_x+high_gradient_y, 
                    M_gradient_x + M_gradient_y, x_loss+ y_loss, path=f'./samples-features/mutual_consist_epoch{hook}.png')
        return mutual_loss

    def reconstruction_error(self, R_low, R_high, I_low_3, I_high_3, L_low, L_high):
        recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 -  L_low))
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - L_high))
        return recon_loss_high + recon_loss_low

    def forward(self, R_low, R_high, I_low, I_high, L_low, L_high, hook=-1):
        I_low_3 = torch.cat([I_low, I_low, I_low], dim=1)
        I_high_3 = torch.cat([I_high, I_high, I_high], dim=1)
        #network output
        recon_loss = self.reconstruction_error(R_low, R_high, I_low_3, I_high_3, L_low, L_high)
        equal_R_loss = self.reflectance_similarity(R_low, R_high)
        i_mutual_loss = self.mutual_consistency(I_low, I_high, hook=hook)
        # ilux_smooth_loss = self.illumination_smoothness(I_low, L_low, hook=hook) + \
        #             self.illumination_smoothness(I_high, L_high, name='high', hook=hook) 

        decom_loss = recon_loss + 0.01 * equal_R_loss + 0.1 * i_mutual_loss # + 0.08 * ilux_smooth_loss

        return decom_loss

if __name__ == "__main__":
    from dataloader import *
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    root_path_train = r'H:\datasets\Low-Light Dataset\KinD++\LOLdataset\our485'
    list_path_train = build_LOLDataset_list_txt(root_path_train)
    Batch_size = 2
    log("Buliding LOL Dataset...")
    dst_train = LOLDataset(root_path_train, list_path_train, transform=None, crop_size=400, to_RAM=True)
    # But when we are training a model, the mean should have another value
    trainloader = DataLoader(dst_train, batch_size = Batch_size)
    for i, data in enumerate(trainloader):
        _, L_high, name = data
        L_gradient_x = gradient(L_high, "x", device='cpu', kernel='sobel')
        epsilon = 0.01*torch.ones_like(L_gradient_x)
        Denominator_x = torch.max(L_gradient_x, epsilon)
        imgs = Denominator_x
        img = imgs[0].numpy()
        sample(img, figure_size=(1,1), img_dim=400)