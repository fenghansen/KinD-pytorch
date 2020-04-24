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
from losses import Decom_Loss
from models import DecomNet
from base_parser import BaseParser
from dataloader import *

class Decom_Trainer(BaseTrainer):
    def train(self):
        print(f'Using device {self.device}')
        self.model.to(device=self.device)
        summary(self.model, input_size=(3, 48, 48))
        # faster convolutions, but more memory
        # cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
        try:
            for iter in range(self.epochs):
                epoch_loss = 0
                idx = 0
                hook_number = -1
                iter_start_time = time.time()
                # with tqdm(total=self.steps_per_epoch) as pbar:
                for L_low_tensor, L_high_tensor, name in self.dataloader:
                    L_low = L_low_tensor.to(self.device)
                    L_high = L_high_tensor.to(self.device)
                    R_low, I_low = self.model(L_low)
                    R_high, I_high = self.model(L_high)
                    if idx % self.print_frequency == 0:
                        hook_number = -1
                    loss = self.loss_fn(R_low, R_high, I_low, I_high, L_low, L_high, hook=hook_number)
                    hook_number = -1
                    if idx % 8 == 0:
                        print(f"iter: {iter}_{idx}\taverage_loss: {loss.item():.6f}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    idx += 1
                    # pbar.update(1)
                    # pbar.set_postfix({'loss':loss.item()})

                if iter % self.print_frequency == 0:
                    self.test(iter, plot_dir='./images/samples-decom')

                if iter % self.save_frequency == 0:
                    torch.save(self.model.state_dict(), './weights/decom_net_test3.pth')
                    log("Weight Has saved as 'decom_net.pth'")
                        
                scheduler.step()
                iter_end_time = time.time()
                log(f"Time taken: {iter_end_time - iter_start_time:.3f} seconds\t lr={scheduler.get_lr()[0]:.6f}")

        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), 'INTERRUPTED_decom.pth')
            log('Saved interrupt_decom')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    @no_grad
    def test(self, epoch=-1, plot_dir='./images/samples-decom'):
        self.model.eval()
        hook = 0
        for L_low_tensor, L_high_tensor, name in self.dataloader_test:
            L_low = L_low_tensor.to(self.device)
            L_high = L_high_tensor.to(self.device)
            R_low, I_low = self.model(L_low)
            R_high, I_high = self.model(L_high)
            
            if epoch % (self.print_frequency*10) == 0:
                loss = self.loss_fn(R_low, R_high, I_low, I_high, L_low, L_high, hook=hook)
                hook += 1
                loss = 0

            R_low_np = R_low.detach().cpu().numpy()[0]
            R_high_np = R_high.detach().cpu().numpy()[0]
            I_low_np = I_low.detach().cpu().numpy()[0]
            I_high_np = I_high.detach().cpu().numpy()[0]
            L_low_np = L_low_tensor.numpy()[0]
            L_high_np = L_high_tensor.numpy()[0]
            sample_imgs = np.concatenate( (R_low_np, I_low_np, L_low_np,
                                        R_high_np, I_high_np, L_high_np), axis=0 )
            filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch}.png')
            split_point = [0, 3, 4, 7, 10, 11, 14]
            img_dim = I_low_np.shape[1:]
            sample(sample_imgs, split=split_point, figure_size=(2, 3), 
                        img_dim=img_dim, path=filepath, num=epoch)


if __name__ == "__main__":
    criterion = Decom_Loss()
    model = DecomNet()

    parser = BaseParser()
    args = parser.parse()
    args.checkpoint = True
    if args.checkpoint is not None:
        pretrain = torch.load('./weights/decom_net.pth')
        model.load_state_dict(pretrain)
        print('Model loaded from decom_net.pth')

    with open(args.config) as f:
        config = yaml.load(f)

    root_path_train = r'H:\datasets\Low-Light Dataset\KinD++\LOLdataset\our485'
    root_path_test = r'C:\DeepLearning\KinD_plus-master\LOLdataset\eval15'
    list_path_train = build_LOLDataset_list_txt(root_path_train)
    list_path_test = build_LOLDataset_list_txt(root_path_test)
    # list_path_train = os.path.join(root_path_train, 'pair_list.csv')
    # list_path_test = os.path.join(root_path_test, 'pair_list.csv')

    log("Buliding LOL Dataset...")
    # transform = transforms.Compose([transforms.ToTensor(),])
    dst_train = LOLDataset(root_path_train, list_path_train,
                            crop_size=config['length'], to_RAM=True)
    dst_test = LOLDataset(root_path_test, list_path_test,
                            crop_size=config['length'], to_RAM=True, training=False)

    train_loader = DataLoader(dst_train, batch_size = config['batch_size'], shuffle=True)
    test_loader = DataLoader(dst_test, batch_size=1)

    # if args.noDecom is True:
    #     root_path_valid = r'H:\datasets\Low-Light Dataset\LOLdataset_decom\our485'
    #     list_path_valid = os.path.join(root_path_test, 'pair_list.csv')

    #     log("Buliding LOL Dataset (noDecom)...")
    #     # transform = transforms.Compose([transforms.ToTensor()])
    #     dst_valid = LOLDataset_Decom(root_path_test, list_path_test,
    #                             crop_size=config['length'], to_RAM=True, training=False)
    #     valid_loader = DataLoader(dst_train, batch_size = config['batch_size'], shuffle=True)

    trainer = Decom_Trainer(config, train_loader, criterion, model, dataloader_test=test_loader)
    # --config ./config/config.yaml
    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test()