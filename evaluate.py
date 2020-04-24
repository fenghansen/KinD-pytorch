import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import yaml
import sys
from tqdm import tqdm
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
from torchvision import transforms
from torchsummary import summary
from base_trainer import BaseTrainer
from losses import *
from models import *
from base_parser import BaseParser
from dataloader import *

class KinD_noDecom_Trainer(BaseTrainer):
    @no_grad
    def test(self, epoch=-1, plot_dir='./images/samples-KinD'):
        self.model.eval()
        self.model.to(device=self.device)
        if 'decom_net' in model._modules:
            for L_low_tensor, L_high_tensor, name in self.dataloader_test:
                L_low = L_low_tensor.to(self.device)
                L_high = L_high_tensor.to(self.device)

                R_low, I_low = self.model.decom_net(L_low)
                R_high, I_high = self.model.decom_net(L_high)
                I_low_3 = torch.cat([I_low, I_low, I_low], dim=1)
                I_high_3 = torch.cat([I_high, I_high, I_high], dim=1)

                output_low = I_low_3 * R_low
                output_high = I_high_3 * R_high

                b = 0.7; w=0.5
                bright_low = torch.mean(I_low)
                # bright_high = torch.mean(I_high)
                bright_high = torch.ones_like(bright_low) * b + bright_low * w
                ratio = torch.div(bright_high, bright_low)
                log(f"Brightness: {bright_high}\tIllumation Magnification: {ratio.item()}")
                # ratio_map = torch.ones_like(I_low) * ratio
                
                R_final, I_final, output_final = self.model(L_low, ratio)

                R_final_np = R_final.detach().cpu().numpy()[0]
                I_final_np = I_final.detach().cpu().numpy()[0]
                R_low_np = R_low.detach().cpu().numpy()[0]
                I_low_np = I_low.detach().cpu().numpy()[0]
                R_high_np = R_high.detach().cpu().numpy()[0]
                I_high_np = I_high.detach().cpu().numpy()[0]
                output_final_np = output_final.detach().cpu().numpy()[0]
                output_low_np = output_low.detach().cpu().numpy()[0]
                output_high_np = output_high.detach().cpu().numpy()[0]
                # ratio_map_np = ratio_map.detach().cpu().numpy()[0]
                L_low_np = L_low_tensor.numpy()[0]
                L_high_np = L_high_tensor.numpy()[0]
                
                sample_imgs = np.concatenate( (R_low_np, I_low_np, output_low_np, L_low_np,
                                            R_high_np, I_high_np, output_high_np, L_high_np,
                                            R_final_np, I_final_np, output_final_np, L_high_np), axis=0 )

                filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch}.png')
                split_point = [0, 3, 4, 7, 10, 13, 14, 17, 20, 23, 24, 27, 30]
                img_dim = I_high_np.shape[1:]
                sample(sample_imgs, split=split_point, figure_size=(3, 4), 
                            img_dim=img_dim, path=filepath, num=epoch)
        else:
            for R_low_tensor, I_low_tensor, R_high_tensor, I_high_tensor, name in self.dataloader_test:
                R_low = R_low_tensor.to(self.device)
                R_high = R_high_tensor.to(self.device)
                I_low = I_low_tensor.to(self.device)
                I_high = I_high_tensor.to(self.device)
                I_high_3 = torch.cat([I_high, I_high, I_high], dim=1)
                output_high = I_high_3 * R_high

                # while True:
                #     b = float(input('请输入增强水平：'))
                #     if b <= 0: break
                b = 0.6; w = 0.5
                bright_low = torch.mean(I_low)
                bright_high = torch.ones_like(bright_low) * b + bright_low * w
                ratio = torch.div(bright_high, bright_low)
                print(bright_high, ratio)
                # ratio_map = torch.ones_like(I_low) * ratio
                
                R_final, I_final, output_final = self.model(R_low, I_low, ratio)

                R_final_np = R_final.detach().cpu().numpy()[0]
                I_final_np = I_final.detach().cpu().numpy()[0]
                output_final_np = output_final.detach().cpu().numpy()[0]
                output_high_np = output_high.detach().cpu().numpy()[0]
                # ratio_map_np = ratio_map.detach().cpu().numpy()[0]
                I_high_np = I_high_tensor.numpy()[0]
                R_high_np = R_high_tensor.numpy()[0]
                
                sample_imgs = np.concatenate( (R_high_np, I_high_np, output_high_np,
                                            R_final_np, I_final_np, output_final_np), axis=0 )

                filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch}.png')
                split_point = [0, 3, 4, 7, 10, 11, 14]
                img_dim = I_high_np.shape[1:]
                sample(sample_imgs, split=split_point, figure_size=(2, 3), 
                            img_dim=img_dim, path=filepath, num=epoch)
            

if __name__ == "__main__":
    criterion = None
    parser = BaseParser()
    args = parser.parse()
    # args.noDecom = True
    with open(args.config) as f:
        config = yaml.load(f)
    if config['noDecom'] is True:
        model = KinD_noDecom()
    else:
        model = KinD()

    if args.checkpoint is not None:
        if config['noDecom'] is False:
            pretrain_decom = torch.load('./weights/decom_net.pth')
            model.decom_net.load_state_dict(pretrain_decom)
            log('Model loaded from decom_net.pth')
        pretrain_resotre = torch.load('./weights/restore_net.pth')
        model.restore_net.load_state_dict(pretrain_resotre)
        log('Model loaded from restore_net.pth')
        pretrain_illum = torch.load('./weights/illum_net.pth')
        model.illum_net.load_state_dict(pretrain_illum)
        log('Model loaded from illum_net.pth')

    if config['noDecom'] is True:
        root_path_test = r'H:\datasets\Low-Light Dataset\LOLdataset_decom\eval15'
        list_path_test = os.path.join(root_path_test, 'pair_list.csv')

        log("Buliding LOL Dataset (noDecom)...")
        # transform = transforms.Compose([transforms.ToTensor()])
        dst_test = LOLDataset_Decom(root_path_test, list_path_test,
                                crop_size=config['length'], to_RAM=True, training=False)
    else:
        root_path_test = r'C:\DeepLearning\KinD_plus-master\LOLdataset\eval15'
        list_path_test = os.path.join(root_path_test, 'pair_list.csv')

        log("Buliding LOL Dataset...")
        # transform = transforms.Compose([transforms.ToTensor()])
        dst_test = LOLDataset(root_path_test, list_path_test,
                                crop_size=config['length'], to_RAM=True, training=False)

    test_loader = DataLoader(dst_test, batch_size=1)

    KinD = KinD_noDecom_Trainer(config, None, criterion, model, dataloader_test=test_loader)
    
    # Please change your output direction here
    output_dir = './images/samples-KinD'
    KinD.test(plot_dir=output_dir)