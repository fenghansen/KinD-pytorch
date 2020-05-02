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

class Restore_Trainer(BaseTrainer):
    def __init__(self, config, dataloader, criterion, model, 
            dataloader_test=None, decom_net=None):
        super().__init__(config, dataloader, criterion, model, dataloader_test)
        log(f'Using device {self.device}')
        self.decom_net = decom_net
        self.decom_net.to(device=self.device)

    def train(self):
        # print(self.model)
        summary(self.model, input_size=[(3, 256, 256), (1,256,256)])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99426) #0.977237, 0.986233
        try:
            for iter in range(self.epochs):
                epoch_loss = 0
                idx = 0
                hook_number = -1
                iter_start_time = time.time()
                # with tqdm(total=self.steps_per_epoch) as pbar:
                for L_low_tensor, L_high_tensor, name in self.dataloader:
                    optimizer.zero_grad()
                    L_low = L_low_tensor.to(self.device)
                    L_high = L_high_tensor.to(self.device)

                    with torch.no_grad():
                        R_low, I_low = self.decom_net(L_low)
                        R_high, I_high = self.decom_net(L_high)
                        
                    R_restore = self.model(R_low, I_low)

                    if idx % self.print_frequency == 0:
                        hook_number = iter
                    loss = self.loss_fn(R_restore, R_high, hook=hook_number)
                    hook_number = -1
                    if idx % 8 == 0:
                        log(f"iter: {iter}_{idx}\taverage_loss: {loss.item():.6f}")
                    loss.backward()
                    optimizer.step()
                    idx += 1
                    # pbar.update(1)
                    # pbar.set_postfix({'loss':loss.item()})

                if iter % self.print_frequency == 0:
                    self.test(iter, plot_dir='./images/samples-restore-MSIA')

                if iter % self.save_frequency == 0:
                    torch.save(self.model.state_dict(), f'./weights/restore_net_MSIA_{iter//100}.pth')
                    log("Weight Has saved as 'restore_net.pth'")
                
                scheduler.step()
                iter_end_time = time.time()
                log(f"Time taken: {iter_end_time - iter_start_time:.3f} seconds\t lr={scheduler.get_lr()[0]:.6f}")
                # print("End of epochs {.0f}, Time taken: {.3f}, average loss: {.5f}".format(
                #                 idx, iter_end_time - iter_start_time, epoch_loss / idx))

        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), './weights/INTERRUPTED_restore.pth')
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    @no_grad
    def test(self, epoch=-1, plot_dir='./images/samples-restore'):
        self.model.eval()
        for L_low_tensor, L_high_tensor, name in self.dataloader_test:
            L_low = L_low_tensor.to(self.device)
            L_high = L_high_tensor.to(self.device)

            R_low, I_low = self.decom_net(L_low)
            R_high, I_high = self.decom_net(L_high)

            R_restore = self.model(R_low, I_low)
        
            R_restore_np = R_restore.detach().cpu().numpy()[0]
            I_low_np = I_low.detach().cpu().numpy()[0]
            R_low_np = R_low.detach().cpu().numpy()[0]
            R_high_np = R_high.detach().cpu().numpy()[0]
            sample_imgs = np.concatenate( (I_low_np, R_low_np, R_restore_np, R_high_np), axis=0 )
            
            filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch//100}.png')
            split_point = [0, 1, 4, 7, 10]
            img_dim = I_low_np.shape[1:]
            sample(sample_imgs, split=split_point, figure_size=(2, 2), 
                        img_dim=img_dim, path=filepath, num=epoch)

if __name__ == "__main__":
    criterion = Restore_Loss()
    model = RestoreNet_MSIA()
    decom_net = DecomNet()

    parser = BaseParser()
    args = parser.parse()

    with open(args.config) as f:
        config = yaml.load(f)
    args.checkpoint = True

    if args.checkpoint is not None:
        if config['noDecom'] is False:
            decom_net = load_weights(decom_net, path='./weights/decom_net.pth')
            log('DecomNet loaded from decom_net.pth')
        model = load_weights(model, path='./weights/restore_net_MSIA_1.pth')
        log('Model loaded from restore_net_MSIA.pth')

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
    # train_loader = data_prefetcher(train_loader)
    test_loader = DataLoader(dst_test, batch_size=1)

    trainer = Restore_Trainer(config, train_loader, criterion, model, 
                            dataloader_test=test_loader, decom_net=decom_net)

    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test()