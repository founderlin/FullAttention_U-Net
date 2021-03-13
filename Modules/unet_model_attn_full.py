# Full Attention U-Net

import torch.nn.functional as F
from Modules.unet_parts_att_full import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.down_sc1to4 = DownSC(64, 512, 8)
        self.down_sc1to3 = DownSC(64, 256, 4)
        self.down_sc1to2 = DownSC(64, 128, 2)
        self.down_sc2to4 = DownSC(128, 512, 4)
        self.down_sc2to3 = DownSC(128, 256, 2)
        self.down_sc3to4 = DownSC(256, 512, 2)
        self.conv_sc4 = nn.Conv2d(2048, 512, kernel_size=3)
        self.conv_sc3 = nn.Conv2d(768, 256, kernel_size=3)
        self.conv_sc2 = nn.Conv2d(256, 128, kernel_size=3)

        self.attn4 = AttnBlock(1024, 512)
        self.attn3 = AttnBlock(512, 256)
        self.attn2 = AttnBlock(256, 128)
        self.attn1 = AttnBlock(128, 64)


    def forward(self, x):
        x1 = self.inc(x)  # 64       
        x2 = self.down1(x1)  # 128  
        x3 = self.down2(x2)  # 256     
        x4 = self.down3(x3)  # 512      
        x5 = self.down4(x4)  # 1024

        x1to4 = self.down_sc1to4(x1)  # 512
        x1to3 = self.down_sc1to3(x1)  # 256        
        x1to2 = self.down_sc1to2(x1)  # 128
        x2to4 = self.down_sc2to4(x2)  # 512
        x2to3 = self.down_sc2to3(x2)  # 256
        x3to4 = self.down_sc3to4(x3)  # 512

        x4a = self.attn4(x4, x5)
        x3to4a = self.attn4(x3to4, x5)
        x2to4a = self.attn4(x2to4, x5)
        x1to4a = self.attn4(x1to4, x5)
        x4 = torch.cat([x4, x1to4a], dim=1)  # 512+512 = 1024
        x4 = torch.cat([x4, x2to4a], dim=1)  # 1024+512 = 1536
        x4 = torch.cat([x4, x3to4a], dim=1)  # 1536+512 = 2048
        x4 = self.conv_sc4(x4)  # 2048 => 512
        x = self.up1(x5, x4)  # 512

        x3a = self.attn3(x3, x)
        x2to3a = self.attn3(x2to3, x)
        x1to3a = self.attn3(x1to3, x)
        x3 = torch.cat([x3, x1to3a], dim=1)  # 256+256 = 512
        x3 = torch.cat([x3, x2to3a], dim=1)  # 512+256 = 768
        x3 = self.conv_sc3(x3)  # 768 => 256
        x = self.up2(x, x3)  # 256

        x2a = self.attn2(x2, x)
        x1to2a = self.attn2(x1to2, x)
        x2 = torch.cat([x2, x1to2a], dim=1)  # 125*2 = 256
        x2 = self.conv_sc2(x2)  # 256 => 128
        x = self.up3(x, x2)  # 128
        
        x1a = self.attn1(x1, x)
        x = self.up4(x, x1a)  # 64

        logits = self.outc(x)
        
        return logits
