# Parts of the Attention U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *

# DoubleConv module: Two consecutive convolution operations.
class DoubleConv(nn.Module):
    # (convolution => [BN] => ReLU) * 2"

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Down module: Downsampling
class Down(nn.Module):
    #nDownscaling with maxpool then double conv

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Up module: Upsampling
class Up(nn.Module):
    # Upscaling then double conv

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        # print(out_channels, in_channels, int(out_channels / 2))
        self.attention = AttentionBlock(out_channels, in_channels, int(out_channels / 2))
        # print("----", in_channels // 2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x_att = self.attention(x2, x1)
        # print("x1", np.shape(x1))
        x1 = self.up(x1)
        # print("x1", np.shape(x1))
        # input is CHW

        diffY = torch.tensor([x_att.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x_att.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x_att, x1], dim=1)
        # print(np.shape(x))
        return self.conv(x)

# OutConv module
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)