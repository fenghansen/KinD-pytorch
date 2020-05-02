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
        torch.backends.cudnn.benchmark = True

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
                idx = 0
                iter_start_time = time.time()
                for L_low_tensor, L_high_tensor, name in self.dataloader:
                    optimizer.zero_grad()
                    L_low = L_low_tensor.to(self.device)
                    L_high = L_high_tensor.to(self.device)

                    with torch.no_grad():
                        _, I_low = self.decom_net(L_low)
                        _, I_high = self.decom_net(L_high)
                    
                    I_out, I_standard = self.model(I_low, 1)
                    loss = self.loss_fn(I_out, I_high, I_standard)

                    if idx % 6 == 0:
                        log(f"iter: {iter}_{idx}\taverage_loss: {loss.item():.6f}")
                    loss.backward()
                    optimizer.step()
                    idx += 1

                if iter % self.print_frequency == 0:
                    self.test(iter, plot_dir='./images/samples-illum-custom')

                if iter % self.save_frequency == 0:
                    torch.save(self.model.state_dict(), f'./weights/illum_net_custom_{iter//100}.pth')
                    log("Weight Has saved as 'illum_net.pth'")
                        
                scheduler.step()
                iter_end_time = time.time()
                w, sigma = self.model.get_parameter()
                log(f"w:{float(w):.4f}\t sigma:{float(sigma):.2f}")
                log(f"Time taken: {iter_end_time - iter_start_time:.3f} seconds\t lr={scheduler.get_lr()[0]:.6f}")

        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), './weights/INTERRUPTED_illum_custom.pth')
            log('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    @no_grad
    def test(self, epoch=-1, plot_dir='./images/samples-illum'):
        self.model.eval()
        for L_low_tensor, L_high_tensor, name in self.dataloader_test:
            L_low = L_low_tensor.to(self.device)
            L_high = L_high_tensor.to(self.device)

            with torch.no_grad():
                _, I_low = self.decom_net(L_low)
                _, I_high = self.decom_net(L_high)
                I_out, I_standard = self.model(I_low, 1)
            # I_low_standard = standard_illum(I_low, w=0.72, gamma=0.53, blur=True)
            # I_high_standard = standard_illum(I_high, w=0.08, gamma=1.34)

            I_standard_np = I_standard.detach().cpu().numpy()[0]
            I_out_np = I_out.detach().cpu().numpy()[0]
            I_low_np = I_low.detach().cpu().numpy()[0]
            I_high_np = I_high.detach().cpu().numpy()[0]
            # I_low_standard = standard_illum(I_low_np, dynamic=3)
            # I_high_standard = standard_illum(I_high_np)

            sample_imgs = np.concatenate( (I_low_np, I_high_np, I_standard_np, I_out_np), axis=0 )

            filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch//100}.png')
            split_point = [0, 1, 2, 3, 4]
            img_dim = I_low_np.shape[1:]
            sample(sample_imgs, split=split_point, figure_size=(2, 2), 
                        img_dim=img_dim, path=filepath, num=epoch)
            

if __name__ == "__main__":
    criterion = Illum_Custom_Loss()
    decom_net = DecomNet()
    model = IllumNet_Custom()

    parser = BaseParser()
    args = parser.parse()

    with open(args.config) as f:
        config = yaml.load(f)

    args.checkpoint = True
    if args.checkpoint is not None:
        if config['noDecom'] is False:
            decom_net = load_weights(decom_net, path='./weights/decom_net.pth')
            log('DecomNet loaded from decom_net.pth')
        model = load_weights(model, path='./weights/illum_net_custom_0.pth')
        log('Model loaded from illum_net.pth')

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