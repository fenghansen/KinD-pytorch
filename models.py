import torch
import torch.nn as nn
import torch.nn.functional as F
from base_layers import *

class DecomNet(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.conv_input = Conv2D(3, filters)
        # top path build Reflectance map
        self.maxpool_r1 = MaxPooling2D()
        self.conv_r1 = Conv2D(filters, filters*2)
        self.maxpool_r2 = MaxPooling2D()
        self.conv_r2 = Conv2D(filters*2, filters*4)
        self.deconv_r1 = ConvTranspose2D(filters*4, filters*2)
        self.concat_r1 = Concat()
        self.conv_r3 = Conv2D(filters*4, filters*2)
        self.deconv_r2 = ConvTranspose2D(filters*2, filters)
        self.concat_r2 = Concat()
        self.conv_r4 = Conv2D(filters*2, filters)
        self.conv_r5 = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self.R_out = nn.Sigmoid()
        # bottom path build Illumination map
        self.conv_i1 = Conv2D(filters, filters)
        self.concat_i1 = Concat()
        self.conv_i2 = nn.Conv2d(filters*2, 1, kernel_size=3, padding=1)
        self.I_out = nn.Sigmoid()

    def forward(self, x):
        conv_input = self.conv_input(x)
        # build Reflectance map
        maxpool_r1 = self.maxpool_r1(conv_input)
        conv_r1 = self.conv_r1(maxpool_r1)
        maxpool_r2 = self.maxpool_r2(conv_r1)
        conv_r2 = self.conv_r2(maxpool_r2)
        deconv_r1 = self.deconv_r1(conv_r2)
        concat_r1 = self.concat_r1(conv_r1, deconv_r1)
        conv_r3 = self.conv_r3(concat_r1)
        deconv_r2 = self.deconv_r2(conv_r3)
        concat_r2 = self.concat_r2(conv_input, deconv_r2)
        conv_r4 = self.conv_r4(concat_r2)
        conv_r5 = self.conv_r5(conv_r4)
        R_out = self.R_out(conv_r5)
        
        # build Illumination map
        conv_i1 = self.conv_i1(conv_input)
        concat_i1 = self.concat_i1(conv_r4, conv_i1)
        conv_i2 = self.conv_i2(concat_i1)
        I_out = self.I_out(conv_i2)

        return R_out, I_out