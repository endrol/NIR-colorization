#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F
import unet_parts
# python 3 confusing imports :(
from unet_part2 import *

'''this part to downsize and add features, used for both original and classification use'''
class downsize_part(nn.Module):
    def __init__(self, n_channels):
        super(downsize_part, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)   # 1/8, 512

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x4

'''classification part. '''
class classification_net(nn.Module):
    def __init__(self):
        super(classification_net, self).__init__()
        # self.add_module(downsize_part.eval())
        # self.down_size = downsize_part.eval()
        self.conv1 = global_conv(512, 512, 256)
        # self.conv2 = linear_op(25088, 1024)
        # self.conv3 = linear_op(1024, 512)

    def forward(self, x):
        # x1 = self.down_size(x)
        x = self.conv1(x)
        # x2 = self.conv2(x1)
        # x = self.conv3(x2)
        return x


class classification_part(nn.Module):
    def __init__(self):
        super(classification_part, self).__init__()


    def forward(self, x):
        return x



'''this part try to produce mid-level feature'''
class mid_level(nn.Module):
    def __init__(self):
        super(mid_level, self).__init__()
        # self.down_size = downsize_part.eval()
        self.conv = mid_level_feature()

    def forward(self, x):
        # x1 = self.down_size(x)
        x = self.conv(x)
        return x


class mid2_feat(nn.Module):
    def __init__(self):
        super(mid2_feat, self).__init__()
        # self.global_part = classification_net.eval()
        # self.conv1 = linear_op(512, 256)    # downsize the channel number
        self.conv = replicate_op()

    def forward(self, x):
        # x1 = self.global_part(x)
        # x1 = self.conv1(x)
        x = self.conv(x)
        return x


class Unet2(nn.Module):
    def __init__(self, in_chan, n_classes, nlabels):
        super(Unet2, self).__init__()

        self.inc = inconv(in_chan, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)

        self.global_feat = classification_net()
        self.mid_level1 = mid_level()
        self.mid_level2 = mid2_feat()

        self.upp = up2(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

        # classification result
        self.classication = classNet(nlabels)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_class = self.classication(x4)

        x5 = self.global_feat(x4)    # 1/64*1/64 256

        x6 = self.mid_level1(x4)   # 1/8*1/8 256
        x7 = self.mid_level2(x5)

        x8 = torch.cat((x7, x6), 1)  # 4 parameters, batch number, channel number, H and W

        # print(x8.size())

        x = self.upp(x8, x4)   # 1024, 1/8 * 1/8
        # print(x.size())
        x = self.up2(x, x3)
        # print(x.size())
        x = self.up3(x, x2)
        # print(x.size())
        x = self.up4(x, x1)
        # print(x.size())
        x = self.outc(x)
        # print(x.size())
        return x, x_class





# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes):  # 定义函数
#         super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)  # 1/16 * 1/16
#         self.up1 = up(1024, 256)   # output 1/8 256
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 64)
#         self.outc = outconv(64, n_classes)
#
#     def forward(self, x):  # 执行函数
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         return x
