#!/usr/bin/python

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


'''
这个部分是网络的组成
基本的一层应该如何操作
'''


class double_conv(nn.Module):     # 各层的神经网络
                                    # Container
    '''(conv => BN => ReLU) * 2'''    # bn -> Batch normalization

    def __init__(self, in_ch, out_ch):    # input channels and output channels
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(             # 依照传入的顺序添加layer 到 容器中(container)

            # 此处加入一个module
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # Conv2d   3 is kernel size
            nn.BatchNorm2d(out_ch),    # batch normalization / without Learnable Parameters
            nn.ReLU(inplace=True),        # in-place operation在原地操作，不使用新的内存
            nn.Conv2d(out_ch, out_ch, 3, padding=1),      # the second time
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):     # 计算前向计算的方法
        x = self.conv(x)     # self.conv(x)实际上为x = self.conv.forward(x)   调用了nn.Conv2d()的forward()函数
        return x


class inconv(nn.Module):    # the first input image.
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):    # maxpooling
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(           # now i think this is just a name
            nn.MaxPool2d(2),      # max pooling 2 is the size
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):    # upsampling
    def __init__(self, in_ch, out_ch, bilinear=True):     # default setting that bilinear is true
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)          # 双线性插值 upsampling
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)    # 逆卷积 or 转置卷积    deconvolution

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, ((diffX + 1) // 2, int(diffX / 2),
                        (diffY + 1) // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),   # 1*1 convolution
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        return x
