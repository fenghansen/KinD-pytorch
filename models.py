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


class IllumNet(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.concat_input = Concat()
        # bottom path build Illumination map
        self.conv_i1 = Conv2D(2, filters)
        self.conv_i2 = Conv2D(filters, filters)
        self.conv_i3 = Conv2D(filters, filters)
        self.conv_i4 = nn.Conv2d(filters, 1, kernel_size=3, padding=1)
        self.I_out = nn.Sigmoid()

    def forward(self, I, ratio):
        with torch.no_grad():
            ratio_map = torch.ones_like(I) * ratio
        concat_input = self.concat_input(I, ratio_map)        
        # build Illumination map
        conv_i1 = self.conv_i1(concat_input)
        conv_i2 = self.conv_i2(conv_i1)
        conv_i3 = self.conv_i3(conv_i2)
        conv_i4 = self.conv_i4(conv_i3)
        I_out = self.I_out(conv_i4)

        return I_out


class RestoreNet_Unet(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.conv1_1 = Conv2D(4, filters)
        self.conv1_2 = Conv2D(filters, filters)
        self.pool1 = MaxPooling2D()
        
        self.conv2_1 = Conv2D(filters, filters*2)
        self.conv2_2 = Conv2D(filters*2, filters*2)
        self.pool2 = MaxPooling2D()
        
        self.conv3_1 = Conv2D(filters*2, filters*4)
        self.conv3_2 = Conv2D(filters*4, filters*4)
        self.pool3 = MaxPooling2D()
        
        self.conv4_1 = Conv2D(filters*4, filters*8)
        self.conv4_2 = Conv2D(filters*8, filters*8)
        self.pool4 = MaxPooling2D()
        
        self.conv5_1 = Conv2D(filters*8, filters*16)
        self.conv5_2 = Conv2D(filters*16, filters*16)
        self.dropout = nn.Dropout2d(0.5)
        
        self.upv6 = ConvTranspose2D(filters*16, filters*8)
        self.concat6 = Concat()
        self.conv6_1 = Conv2D(filters*16, filters*8)
        self.conv6_2 = Conv2D(filters*8, filters*8)
        
        self.upv7 = ConvTranspose2D(filters*8, filters*4)
        self.concat7 = Concat()
        self.conv7_1 = Conv2D(filters*8, filters*4)
        self.conv7_2 = Conv2D(filters*4, filters*4)
        
        self.upv8 = ConvTranspose2D(filters*4, filters*2)
        self.concat8 = Concat()
        self.conv8_1 = Conv2D(filters*4, filters*2)
        self.conv8_2 = Conv2D(filters*2, filters*2)
        
        self.upv9 = ConvTranspose2D(filters*2, filters)
        self.concat9 = Concat()
        self.conv9_1 = Conv2D(filters*2, filters)
        self.conv9_2 = Conv2D(filters, filters)
        
        self.conv10_1 = nn.Conv2d(filters, 3, kernel_size=1, stride=1)
        self.out = nn.Sigmoid()
    
    def forward(self, R, I):
        x = torch.cat([R, I], dim=1)
        conv1 = self.conv1_1(x)
        conv1 = self.conv1_2(conv1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = self.pool1(conv2)
        
        conv3 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv3)
        pool3 = self.pool1(conv3)
        
        conv4 = self.conv4_1(pool3)
        conv4 = self.conv4_2(conv4)
        pool4 = self.pool1(conv4)
        
        conv5 = self.conv5_1(pool4)
        conv5 = self.conv5_2(conv5)
        
        # d = self.dropout(conv5)
        up6 = self.upv6(conv5)
        up6 = self.concat6(conv4, up6)
        conv6 = self.conv6_1(up6)
        conv6 = self.conv6_2(conv6)
        
        up7 = self.upv7(conv6)
        up7 = self.concat7(conv3, up7)
        conv7 = self.conv7_1(up7)
        conv7 = self.conv7_2(conv7)
        
        up8 = self.upv8(conv7)
        up8 = self.concat8(conv2, up8)
        conv8 = self.conv8_1(up8)
        conv8 = self.conv8_2(conv8)
        
        up9 = self.upv9(conv8)
        up9 = self.concat9(conv1, up9)
        conv9 = self.conv9_1(up9)
        conv9 = self.conv9_2(conv9)
        
        conv10 = self.conv10_1(conv9)
        out = self.out(conv10)
        return out

class KinD_noDecom(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        # self.decom_net = DecomNet()
        self.restore_net = RestoreNet_Unet()
        self.illum_net = IllumNet()
    
    def forward(self, R, I, ratio):
        I_final = self.illum_net(I, ratio)
        R_final = self.restore_net(R, I)
        I_final_3 = torch.cat([I_final, I_final, I_final], dim=1)
        output = I_final_3 * R_final
        return R_final, I_final, output


class KinD(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.decom_net = DecomNet()
        self.restore_net = RestoreNet_Unet()
        self.illum_net = IllumNet()
        self.KinD_noDecom = KinD_noDecom()
        self.KinD_noDecom.restore_net = self.restore_net
        self.KinD_noDecom.illum_net = self.illum_net
    
    def forward(self, L, ratio):
        R, I = self.decom_net(L)
        R_final, I_final, output = self.KinD_noDecom(R, I, ratio)
        # I_final = self.illum_net(I, ratio)
        # R_final = self.restore_net(R, I)
        # I_final_3 = torch.cat([I_final, I_final, I_final], dim=1)
        # output = I_final_3 * R_final
        return R_final, I_final, output
