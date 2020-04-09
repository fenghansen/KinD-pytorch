import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gradient(map, direction):
    smooth_kernel_x = torch.Tensor(np.array([[0, 0], [-1, 1]])).view(2, 2, 1, 1)
    smooth_kernel_y = smooth_kernel_x.permute(1, 0, 2, 3)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(nn.Conv2d(3, 3, kernel, strides=1, padding=(1,0))(map))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm

class Decom_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def reflectance_similarity(self, R_low, R_high):
        return F.mse_loss(R_low, R_high)
    
    def illumination_smoothness(self, I, L):
        # L_transpose = L.permute(0, 2, 3, 1)
        # L_gray_transpose = 0.299*L[:,:,:,0] + 0.587*L[:,:,:,1] + 0.114*L[:,:,:,2]
        # L_gray = L.permute(0, 3, 1, 2)
        L_gray = 0.299*L[:,0,:,:] + 0.587*L[:,1,:,:] + 0.114*L[:,2,:,:]
        print(L_gray.size)
        I_gradient_x = gradient(I, "x")
        L_gradient_x = gradient(L_gray, "x")
        x_loss = torch.abs(torch.div(I_gradient_x, torch.max(L_gradient_x, 0.01)))
        I_gradient_y = gradient(I, "y")
        L_gradient_y = gradient(L_gray, "y")
        y_loss = torch.abs(torch.div(I_gradient_y, torch.max(L_gradient_y, 0.01)))
        mut_loss = torch.mean(x_loss + y_loss) 
        return mut_loss
    
    def mutual_consistency(self, I_low, I_high):
        low_gradient_x = gradient(I_low, "x")
        high_gradient_x = gradient(I_high, "x")
        M_gradient_x = low_gradient_x + high_gradient_x
        x_loss = M_gradient_x * torch.exp(-10 * M_gradient_x)
        low_gradient_y = gradient(I_low, "y")
        high_gradient_y = gradient(I_high, "y")
        M_gradient_y = low_gradient_y + high_gradient_y
        y_loss = M_gradient_y * torch.exp(-10 * M_gradient_y)
        mutual_loss = torch.mean(x_loss + y_loss) 
        return mutual_loss

    def reconstruction_error(self, R_low, R_high, I_low_3, I_high_3, L_low, L_high):
        recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 -  L_low))
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - L_high))
        return recon_loss_high + recon_loss_low

    def forward(self, R_low, R_high, I_low, I_high, L_low, L_high):
        I_low_3 = torch.cat([I_low, I_low, I_low])
        I_high_3 = torch.cat([I_high, I_high, I_high])
        #network output

        recon_loss = self.reconstruction_error(R_low, R_high, I_low_3, I_high_3, L_low, L_high)
        equal_R_loss = self.reflectance_similarity(R_low, R_high)
        i_mutual_loss = self.mutual_consistency(L_low, L_high)
        i_input_mutual_loss = self.illumination_smoothness(I_low, L_low) + /
                                self.illumination_smoothness(I_high, L_high) 

        decom_loss = recon_loss + 0.009*equal_R_loss + 0.2*i_mutual_loss + 0.15* i_input_mutual_loss
        return decom_loss