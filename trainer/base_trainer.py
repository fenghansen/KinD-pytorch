import torch
from torch import optim
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

class BaseTrainer:
    def __init__(self, config, dataloader, criterion, model):
        self.initialize(config)
        self.dataloader = dataloader
        self.loss_fn = criterion
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize(self, config):
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.print_frequency = config['print_frequency']
        self.save_frequency = config['save_frequency']
        self.weights_dir = config['weights_dir']
        self.samples_dir = config['samples_dir']           # './logs/samples'
        self.learning_rate = config['learning_rate']

    def train(self):
        print(f'Using device {self.device}')
        summary(self.model, input_size=(3, 48, 48))

        self.model.to(device=self.device)
        # faster convolutions, but more memory
        # cudnn.benchmark = True

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        try:
            for iter in range(self.epochs):
                epoch_loss = 0
                steps = 0
                iter_start_time = time.time()
                for idx, data in enumerate(self.dataloader):
                    input_ = data['input']
                    input_ = input_.to(self.device)
                    target = data['target']
                    target = target.to(self.device)
                    y_pred = self.model(input_)
                    loss = self.loss_fn(y_pred, target)
                    print("iter:  ", idx, "average_loss:  ", loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    steps += 1
                    if idx > 0 and idx % self.save_frequency == 0:
                        # torch.save(self.model.state_dict(), './checkpoints/g_net_{}.pth'.format(str(idx % 3)))
                        print('Saved model.')
                        self.test(iter, idx, plotImage=True, saveImage=True)
                iter_end_time = time.time()
                print("End of epochs {},    Time taken: {},\
                        average loss: {}".format(iter, iter_end_time - iter_start_time, epoch_loss / steps))
                iter_end_time = time.time()
                print("End of epochs {},    Time taken: {.3f},\
                    average loss: {.5f}".format(iter, iter_end_time - iter_start_time, epoch_loss / steps))
        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), 'INTERRUPTED.pth')
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    def test(self, epoch, iter, plotImage=False, saveImage=False):
        pass

    def sample(self, imgs, split=None ,figure_size=(2, 3), img_dim=96, path=None, num=0):
        h, w = figure_size
        if split is None:
            split = range(len(imgs)+1)
        figure = np.zeros((img_dim*h, img_dim*w, 3))
        for i in range(h):
            for j in range(w):
                idx = i*w+j
                if idx >= len(split)-1: break
                digit = imgs[ split[idx] : split[idx+1] ]
                if len(digit) == 1:
                    for k in range(3):
                        figure[i*img_dim: (i+1)*img_dim,
                            j*img_dim: (j+1)*img_dim, k] = digit
                elif len(digit) == 3:
                    for k in range(3):
                        figure[i*img_dim: (i+1)*img_dim,
                            j*img_dim: (j+1)*img_dim, k] = digit[2-k]
        if path is None:
            cv2.imshow('Figure%d'%num, figure)
            cv2.waitKey()
        else:
            figure *= 255
            print(">> Saving Image at {}".format(path))
            cv2.imwrite(path, figure)