import torch
import torch.nn as nn
import torch.nn.functional as F
import unet_model
from unet_model import UNet


class disloss(nn.Module):
    def __init__(self):
        super(disloss, self).__init__()
        self.d = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(64,64,3,1,1,1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            # nn.Sigmoid()
            nn.Tanh()
            # nn.ConvTranspose2d(64, 2, 3, stride=2,padding=1,output_padding=1),
            # nn.BatchNorm2d(2),
            # nn.LogSoftmax(dim=1),

        )

    def forward(self, x1, x2):
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, ((diffY + 1) // 2, int(diffY / 2),
                        (diffX + 1) // 2, int(diffX / 2)))
        diffx = x1 - x2

        x = self.d(diffx)
        return x


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.g = UNet(1, 3)
        self.d = disloss()

    def forward(self,  x1, x2):
        xg = self.g(x1)
        x = self.d(xg, x2)
        return x, xg
