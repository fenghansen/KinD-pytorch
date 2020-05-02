import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import time
import yaml
import sys
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision import transforms
from torchsummary import summary
from base_trainer import BaseTrainer
from losses import *
from models import *
from base_parser import BaseParser
from dataloader import *

class Illum_Trainer(BaseTrainer):
    def __init__(self, config, dataloader, criterion, model, 
            dataloader_test=None, decom_net=None):
        super().__init__(config, dataloader, criterion, model, dataloader_test)
        log(f'Using device {self.device}')
        self.decom_net = decom_net
        self.decom_net.to(device=self.device)

    def train(self):
        self.model.train()
        log(f'Using device {self.device}')
        self.model.to(device=self.device)
        print(self.model)
        # summary(self.model, input_size=[(1, 384, 384), (1,)], batch_size=4)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99426)
        try:
            for iter in range(self.epochs):
                epoch_loss = 0
                idx = 0
                hook_number = -1
                iter_start_time = time.time()
                if self.noDecom is True:
                    # with tqdm(total=self.steps_per_epoch) as pbar:
                    for R_low_tensor, I_low_tensor, R_high_tensor, I_high_tensor, name in self.dataloader:
                        optimizer.zero_grad()
                        I_low = I_low_tensor.to(self.device)
                        I_high = I_high_tensor.to(self.device)
                        with torch.no_grad():
                            ratio_high2low = torch.mean(torch.div((I_low + 0.0001), (I_high + 0.0001)))
                            ratio_low2high = torch.mean(torch.div((I_high + 0.0001), (I_low + 0.0001)))
                        
                        I_low2high_map = self.model(I_low, ratio_low2high)
                        I_high2low_map = self.model(I_high, ratio_high2low)

                        if idx % self.print_frequency == 0:
                            hook_number = iter
                        loss = self.loss_fn(I_low2high_map, I_high, hook=hook_number) + self.loss_fn(I_high2low_map, I_low, hook=hook_number)
                        hook_number = -1
                        if idx % 30 == 0:
                            log(f"iter: {iter}_{idx}\taverage_loss: {loss.item():.6f}")
                            print(ratio_high2low, ratio_low2high)
                        loss.backward()
                        optimizer.step()
                        idx += 1
                        # pbar.update(1)
                        # pbar.set_postfix({'loss':loss.item()})
                else:
                    # with tqdm(total=self.steps_per_epoch) as pbar:
                    for L_low_tensor, L_high_tensor, name in self.dataloader:
                        optimizer.zero_grad()
                        L_low = L_low_tensor.to(self.device)
                        L_high = L_high_tensor.to(self.device)

                        with torch.no_grad():
                            R_low, I_low = self.decom_net(L_low)
                            R_high, I_high = self.decom_net(L_high)
                            # ratio_high2low = torch.mean(torch.div((I_low + 0.0001), (I_high + 0.0001)))
                            # ratio_low2high = torch.mean(torch.div((I_high + 0.0001), (I_low + 0.0001)))
                            bright_low = torch.mean(I_low)
                            bright_high = torch.mean(I_high)
                            ratio_high2low = torch.div(bright_low, bright_high)
                            ratio_low2high = torch.div(bright_high, bright_low)
                        
                        I_low2high_map = self.model(I_low, ratio_low2high)
                        I_high2low_map = self.model(I_high, ratio_high2low)

                        loss = self.loss_fn(I_low2high_map, I_high, hook=hook_number) + \
                                self.loss_fn(I_high2low_map, I_low, hook=hook_number)

                        if idx % 30 == 0:
                            log(f"iter: {iter}_{idx}\taverage_loss: {loss.item():.6f}")
                            print(ratio_high2low, ratio_low2high)
                        loss.backward()
                        optimizer.step()
                        idx += 1
                        # pbar.update(1)
                        # pbar.set_postfix({'loss':loss.item()})

                if iter % self.print_frequency == 0:
                    self.test(iter, plot_dir='./images/samples-illum')

                if iter % self.save_frequency == 0:
                    torch.save(self.model.state_dict(), './weights/illum_net.pth')
                    log("Weight Has saved as 'illum_net.pth'")
                        
                scheduler.step()
                iter_end_time = time.time()
                log(f"Time taken: {iter_end_time - iter_start_time:.3f} seconds\t lr={scheduler.get_lr()[0]:.6f}")

        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), './weights/INTERRUPTED_illum.pth')
            log('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    @no_grad
    def test(self, epoch=-1, plot_dir='./images/samples-illum'):
        self.model.eval()
        if self.noDecom:
            for R_low_tensor, I_low_tensor, R_high_tensor, I_high_tensor, name in self.dataloader_test:
                I_low = I_low_tensor.to(self.device)
                I_high = I_high_tensor.to(self.device)

                ratio_high2low = torch.mean(torch.div((I_low + 0.0001), (I_high + 0.0001)))
                ratio_low2high = torch.mean(torch.div((I_high + 0.0001), (I_low + 0.0001)))
                print(ratio_low2high)
                # 采用粗略的亮度水平估计
                bright_low = torch.mean(I_low)
                bright_high = torch.ones_like(bright_low) * 0.3 + bright_low * 0.55
                ratio_high2low = torch.div(bright_low, bright_high)
                ratio_low2high = torch.div(bright_high, bright_low)
                print(ratio_low2high)

                I_low2high_map = self.model(I_low, ratio_low2high)
                I_high2low_map = self.model(I_high, ratio_high2low)

                I_low2high_np = I_low2high_map.detach().cpu().numpy()[0]
                I_high2low_np = I_high2low_map.detach().cpu().numpy()[0]
                I_low_np = I_low_tensor.numpy()[0]
                I_high_np = I_high_tensor.numpy()[0]
                sample_imgs = np.concatenate( (I_low_np, I_high_np, I_high2low_np, I_low2high_np), axis=0 )

                filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch}.png')
                split_point = [0, 1, 2, 3, 4]
                img_dim = I_low_np.shape[1:]
                sample(sample_imgs, split=split_point, figure_size=(2, 2), 
                            img_dim=img_dim, path=filepath, num=epoch)
        else:
            for L_low_tensor, L_high_tensor, name in self.dataloader_test:
                L_low = L_low_tensor.to(self.device)
                L_high = L_high_tensor.to(self.device)

                R_low, I_low = self.decom_net(L_low)
                R_high, I_high = self.decom_net(L_high)
                bright_low = torch.mean(I_low)
                bright_high = torch.mean(I_high)
                ratio_high2low = torch.div(bright_low, bright_high)
                ratio_low2high = torch.div(bright_high, bright_low)
                print(ratio_low2high)

                I_low2high_map = self.model(I_low, ratio_low2high)
                I_high2low_map = self.model(I_high, ratio_high2low)

                I_low2high_np = I_low2high_map.detach().cpu().numpy()[0]
                I_high2low_np = I_high2low_map.detach().cpu().numpy()[0]
                I_low_np = I_low.detach().cpu().numpy()[0]
                I_high_np = I_high.detach().cpu().numpy()[0]
                sample_imgs = np.concatenate( (I_low_np, I_high_np, I_high2low_np, I_low2high_np), axis=0 )

                filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch}.png')
                split_point = [0, 1, 2, 3, 4]
                img_dim = I_low_np.shape[1:]
                sample(sample_imgs, split=split_point, figure_size=(2, 2), 
                            img_dim=img_dim, path=filepath, num=epoch)

if __name__ == "__main__":
    criterion = Illum_Loss()
    decom_net = DecomNet()
    model = IllumNet()

    parser = BaseParser()
    args = parser.parse()

    with open(args.config) as f:
        config = yaml.load(f)

    args.checkpoint = True
    if args.checkpoint is not None:
        if config['noDecom'] is False:
            decom_net = load_weights(decom_net, path='./weights/decom_net.pth')
            log('DecomNet loaded from decom_net.pth')
        model = load_weights(model, path='./weights/illum_net.pth')
        log('Model loaded from illum_net.pth')

    if config['noDecom'] is True:
        root_path_train = r'H:\datasets\Low-Light Dataset\LOLdataset_decom\our485'
        root_path_test = r'H:\datasets\Low-Light Dataset\LOLdataset_decom\eval15'
        list_path_train = build_LOLDataset_Decom_list_txt(root_path_train)
        list_path_test = build_LOLDataset_Decom_list_txt(root_path_test)
        # list_path_train = os.path.join(root_path_train, 'pair_list.csv')
        # list_path_test = os.path.join(root_path_test, 'pair_list.csv')

        log("Buliding LOL Dataset...")
        # transform = transforms.Compose([transforms.ToTensor()])
        dst_train = LOLDataset_Decom(root_path_train, list_path_train,
                                crop_size=config['length'], to_RAM=True)
        dst_test = LOLDataset_Decom(root_path_test, list_path_test,
                                crop_size=config['length'], to_RAM=True, training=False)

        train_loader = DataLoader(dst_train, batch_size = config['batch_size'], shuffle=True)
        test_loader = DataLoader(dst_test, batch_size=1)

    else:
        root_path_train = r'C:\DeepLearning\KinD_plus-master\LOLdataset\our485'
        root_path_test = r'C:\DeepLearning\KinD_plus-master\LOLdataset\eval15'
        list_path_train = build_LOLDataset_list_txt(root_path_train)
        list_path_test = build_LOLDataset_list_txt(root_path_test)
        # list_path_train = os.path.join(root_path_train, 'pair_list.csv')
        # list_path_test = os.path.join(root_path_test, 'pair_list.csv')

        log("Buliding LOL Dataset...")
        # transform = transforms.Compose([transforms.ToTensor()])
        dst_train = LOLDataset(root_path_train, list_path_train,
                                crop_size=config['length'], to_RAM=True)
        dst_test = LOLDataset(root_path_test, list_path_test,
                                crop_size=config['length'], to_RAM=True, training=False)

        train_loader = DataLoader(dst_train, batch_size = config['batch_size'], shuffle=True)
        test_loader = DataLoader(dst_test, batch_size=1)

    trainer = Illum_Trainer(config, train_loader, criterion, model, 
                            dataloader_test=test_loader, decom_net=decom_net)

    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test()