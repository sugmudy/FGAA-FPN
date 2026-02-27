# Copyright (c) OpenMMLab. All rights reserved.
# FGFM modules.

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule


class FGSegHead(BaseModule):

    def __init__(self,
                 in_channels: int,
                 mid_channels: int = 64,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.conv2 = ConvModule(
            mid_channels,
            1,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None,
            inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x

class FGWeightNet(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(
            num_groups=32 if channels >= 32 else 1,
            num_channels=channels
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)

        self.sharp_k = nn.Parameter(torch.tensor(0.0))
        self.sharp_b = nn.Parameter(torch.tensor(0.0))
        self.lambda_ = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, M):

        k = torch.clamp(self.sharp_k, -10.0, 10.0)
        b = torch.clamp(self.sharp_b, -0.4, 0.4)
        lam = torch.clamp(self.lambda_, 0.0, 1.0)

        M_centered = M - (0.5 + b)
        M_sharp = torch.sigmoid(k * M_centered)

        M_mix = M + lam * (M_sharp - M)

        feat = torch.cat([x, M_mix], dim=1)
        w = self.conv1(feat)
        w = self.relu(self.gn(w))
        w = self.conv2(w)
        M_prime = torch.sigmoid(w)
        return M_prime
