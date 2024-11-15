#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):  # 차원=해상도(HxW)은 낮추고, Channel은 늘려가면서 특징 추출
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # in_channel (ex. RGB=3) -> out_channel,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)  # conv로 feature 뽑고, channel 변경
        pool = self.pool(down)  # pooling으로 해상도 감소

        return pool

class UpSample(nn.Module): # 차원=해상도(HxW)를 복원하고, Channel도 복원 -> 최종에서는 이미지와 같이 HxWxC 가 되어야하니까
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # deconv 실행 또는 interpolation으로 해상도만 키워도 됨(bilinear=True) - 구현 필요
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # in_channel (ex. RGB=3) -> out_channel,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # x1은 upsampling되는 것의 feature map
        # x2는 downsampling에서 upsampling으로 skip connection되는 같은 해상도의 downsampling 단계의 feature map
        x1=self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2,x1],1)  # upsample된 feature map과 downsample에서 온 feature map concat
        x = self.conv(x)

        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)