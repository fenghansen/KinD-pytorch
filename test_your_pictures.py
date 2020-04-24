import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import yaml
import sys
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
from torchvision import transforms
from torchsummary import summary
from base_trainer import BaseTrainer
from losses import *
from models import *
from base_parser import BaseParser
from dataloader import *

class KinD_Player(BaseTrainer):
    def __init__(self, model, dataloader_test, plot_more=False):
        self.dataloader_test = dataloader_test
        self.model = model
        self.plot_more = plot_more
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)

    @no_grad
    def test(self, target_b=0.70, plot_dir='./images/samples-KinD'):
        self.model.eval()
        self.model.to(device=self.device)
        for L_low_tensor, name in self.dataloader_test:
            L_low = L_low_tensor.to(self.device)

            if self.plot_more:
                # Use DecomNet to decomposite Reflectance Map and Illumation Map
                R_low, I_low = self.model.decom_net(L_low)
                # Compute brightness ratio
                bright_low = torch.mean(I_low)
            else:
                bright_low = torch.mean(L_low)
            
            bright_high = torch.ones_like(bright_low) * target_b + 0.5 * bright_low
            ratio = torch.div(bright_high, bright_low)
            log(f"Brightness: {bright_high:.4f}\tIllumation Magnification: {ratio.item():.3f}")
            
            R_final, I_final, output_final = self.model(L_low, ratio)

            output_final_np = output_final.detach().cpu().numpy()[0]
            L_low_np = L_low_tensor.numpy()[0]
            # Only plot result 
            filepath = os.path.join(plot_dir, f'{name[0]}.png')
            split_point = [0, 3]
            img_dim = L_low_np.shape[1:]
            sample(output_final_np, split=split_point, figure_size=(1, 1), 
                        img_dim=img_dim, path=filepath)

            if self.plot_more:
                R_final_np = R_final.detach().cpu().numpy()[0]
                I_final_np = I_final.detach().cpu().numpy()[0]
                R_low_np = R_low.detach().cpu().numpy()[0]
                I_low_np = I_low.detach().cpu().numpy()[0]
                
                sample_imgs = np.concatenate( (R_low_np, I_low_np, L_low_np,
                                            R_final_np, I_final_np, output_final_np), axis=0 )
                filepath = os.path.join(plot_dir, f'{name[0]}_extra.png')
                split_point = [0, 3, 4, 7, 10, 11, 14]
                img_dim = L_low_np.shape[1:]
                sample(sample_imgs, split=split_point, figure_size=(2, 3), 
                            img_dim=img_dim, path=filepath)
                        
            
class TestParser(BaseParser):
    def parse(self):
        self.parser.add_argument("-p", "--plot_more", default=True,
                                help="Plot intermediate variables. such as R_images and I_images")
        self.parser.add_argument("-c", "--checkpoint", default="./weights/", 
                                help="Path of checkpoints")
        self.parser.add_argument("-i", "--input_dir", default="./images/inputs/", 
                                help="Path of input pictures")
        self.parser.add_argument("-o", "--output_dir", default="./images/outputs/", 
                                help="Path of output pictures")
        self.parser.add_argument("-b", "--b_target", default=0.75, help="Target brightness")
        # self.parser.add_argument("-u", "--use_gpu", default=True, 
        #                         help="If you want to use GPU to accelerate")
        return self.parser.parse_args()


if __name__ == "__main__":
    model = KinD()
    parser = TestParser()
    args = parser.parse()

    input_dir = args.input_dir
    output_dir = args.output_dir
    plot_more = args.plot_more
    checkpoint = args.checkpoint
    decom_net_dir = os.path.join(checkpoint, "decom_net.pth")
    restore_net_dir = os.path.join(checkpoint, "restore_net.pth")
    illum_net_dir = os.path.join(checkpoint, "illum_net.pth")
    
    pretrain_decom = torch.load(decom_net_dir)
    model.decom_net.load_state_dict(pretrain_decom)
    log('Model loaded from decom_net.pth')
    pretrain_resotre = torch.load(restore_net_dir)
    model.restore_net.load_state_dict(pretrain_resotre)
    log('Model loaded from restore_net.pth')
    pretrain_illum = torch.load(illum_net_dir)
    model.illum_net.load_state_dict(pretrain_illum)
    log('Model loaded from illum_net.pth')

    log("Buliding Dataset...")
    dst = CustomDataset(input_dir)
    log(f"There are {len(dst)} images in the input direction...")
    dataloader = DataLoader(dst, batch_size=1)

    KinD = KinD_Player(model, dataloader, plot_more=plot_more)
    
    KinD.test(plot_dir=output_dir, target_b=args.b_target)