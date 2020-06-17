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
        a=torch.tensor()

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

class up2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up2, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
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

'''middle process to apply nn.View in pytorch'''
class Reshape(nn.Module):
    def __init__(self, out_dim):
        super(Reshape, self).__init__()
        self.shape = out_dim

    def forward(self, x):
        return x.view(-1, self.shape)


'''this part handle the classification part in loss function'''
class classNet(nn.Module):
    def __init__(self, nLabels):
        super(classNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.fc1 = nn.Linear(512*15*15, 1024)
        self.re = nn.ReLU(True)
        self.fc2 = nn.Linear(1024, 512)

        self.fc3 = nn.Linear(512, 512)
        self.drop = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, nLabels)
        # self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512*15*15)
        x = self.fc1(x)
        x = self.re(x)
        x = self.fc2(x)
        x = self.re(x)
        x = self.fc3(x)
        x = self.re(x)
        x = self.drop(x)
        x = self.fc4(x)
        # x = self.softmax(x)
        return x


class global_conv(nn.Module):
    def __init__(self, in_ch, out_ch, final_ch):
        super(global_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),       # after this, it's 1/32*1/32, 512 channels  1/32 in enough
            # Reshape(view_out)
            #
            # # new structure
            nn.Conv2d(in_ch, final_ch, 3, stride=1, padding=1),   # 1/32*1/32, 256 channel
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class mid_level_feature(nn.Module):
    def __init__(self):
        super(mid_level_feature, self).__init__()
        self.oper = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.oper(x)
        return x


'''part for linear operation'''
class linear_op(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(linear_op, self).__init__()
        self.oper = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.oper(x)
        return x


class replicate_op(nn.Module):
    def __init__(self):
        super(replicate_op, self).__init__()

    def forward(self, x):
        x1 = torch.cat((x, x, x, x), 2)   # change the feature size
        x = torch.cat((x1, x1, x1, x1), 3)
        return x

# class classification_op(nn.Module):
#     def __init__(self, in_ch, nLabels):
#         super(classification_op, self).__init__()

