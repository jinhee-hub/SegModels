#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet.unet_blocks import DoubleConv, DownSample, UpSample, OutConv

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()    # 부모 클래스의 init 실행: 여기선 nn.module의 init 실헹
        self.inc = DoubleConv(in_channels, 64)
        self.downsample1 = DownSample(64, 128)
        self.downsample2 = DownSample(128, 256)
        self.downsample3 = DownSample(256, 512)
        self.downsample4 = DownSample(512, 1024)

        self.upsample1 = UpSample(1024, 512)
        self.upsample2 = UpSample(512, 256)
        self.upsample3 = UpSample(256, 128)
        self.upsample4 = UpSample(128, 64)

        self.out = OutConv(in_channels=64, out_channels=num_classes)

    def forward(self, x):
        # down1 ~ down4는 down sampling된 결과 feature map (Encoder)
        # x= (b, 3, 512,512)
        inc = self.inc(x)  # down1 = (b,64,512,512)
        down1 = self.downsample1(inc)        # down1 = (b,128,256,256)
        down2 = self.downsample2(down1)    # down1 = (b,256,128,128)
        down3 = self.downsample3(down2)    # down1 = (b,512,64,64)
        down4 = self.downsample4(down3)    # down1 = (b,1024,32,32)

        # up1 ~ up4은 up sampling된 결과 feature map (Decoder)
        up1 = self.upsample1(down4, down3)    # up1 = (b,512,64,64)  # down4 앞단인 bottle neck의 결과 feature map, down3는 같은 해상도의 downsample skip connection
        up2 = self.upsample2(up1, down2)  # up2 = (b,256,128,128) up1은 앞단의 feature map, down2는 같은 해상도의 downsample skip connection
        up3 = self.upsample3(up2, down1)  # up3 = (b,128,256,256) up2는 앞단의 feature map, down1는 같은 해상도의 downsample skip connection
        up4 = self.upsample4(up3, inc)  # up4 = (b,64,512,512) up3는 앞단의 feature map, inc은 같은 해상도의 downsample skip connection

        # 앞에서 산출한 결과 feature map으로 최종 output class 수 만큼의 feature map으로 산출
        out = self.out(up4) #  out = (4b,3,512,512)U-shape(encoder-decoder)의 최종 output인 up4

        return out