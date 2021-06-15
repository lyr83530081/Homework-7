""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 20)
        self.down1 = Down(20, 40)
        self.down2 = Down(40, 80)
        self.down3 = Down(80, 160)
        factor = 2 if bilinear else 1
        self.down4 = Down(160, 320 // factor)
        self.up1 = Up(320, 160 // factor, bilinear)
        self.up2 = Up(160, 80 // factor, bilinear)
        self.up3 = Up(80, 40 // factor, bilinear)
        self.up4 = Up(40, 20, bilinear)
        self.outc = OutConv(20, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
