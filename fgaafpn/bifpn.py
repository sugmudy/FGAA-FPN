# Copyright (c) OpenMMLab. All rights reserved.
# BiFPN modules.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule


class DepthwiseSeparableConv(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.dw_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.pw_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw_conv(self.dw_conv(x))


class BiFPNBlock(BaseModule):
    def __init__(self,
                 in_channels: int,
                 num_outs: int,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 use_separable_conv: bool = True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_outs = num_outs
        self.in_channels = in_channels

        if use_separable_conv:
            conv_factory = lambda ic, oc: DepthwiseSeparableConv(
                ic, oc, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        else:
            conv_factory = lambda ic, oc: ConvModule(
                ic,
                oc,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                inplace=False)

        self.td_convs = nn.ModuleList([conv_factory(in_channels, in_channels) for _ in range(num_outs)])
        self.bu_convs = nn.ModuleList([conv_factory(in_channels, in_channels) for _ in range(num_outs)])

        self.w_td = nn.ParameterList([nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(num_outs - 1)])
        self.w_bu = nn.ParameterList([nn.Parameter(torch.ones(2, dtype=torch.float32))])
        for _ in range(1, num_outs):
            self.w_bu.append(nn.Parameter(torch.ones(3, dtype=torch.float32)))

        for w in list(self.w_td) + list(self.w_bu):
            nn.init.constant_(w, 1.0)

        self.eps = 1e-4

    @staticmethod
    def _normalize_weights(w: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        w = F.relu(w)
        return w / (torch.sum(w) + eps)

    @staticmethod
    def _swish(x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(feats) == self.num_outs
        inputs = feats
        num_levels = self.num_outs

        td_feats = [None for _ in range(num_levels)]
        top_idx = num_levels - 1
        td_feats[top_idx] = self.td_convs[top_idx](self._swish(inputs[top_idx]))

        for i in range(num_levels - 2, -1, -1):
            cur = inputs[i]
            up = F.interpolate(td_feats[i + 1], size=cur.shape[-2:], mode='bilinear', align_corners=False)
            w = self._normalize_weights(self.w_td[i], self.eps)
            fused = self._swish(w[0] * cur + w[1] * up)
            td_feats[i] = self.td_convs[i](fused)

        bu_feats = [None for _ in range(num_levels)]
        w0 = self._normalize_weights(self.w_bu[0], self.eps)
        bu_feats[0] = self.bu_convs[0](self._swish(w0[0] * inputs[0] + w0[1] * td_feats[0]))

        for i in range(1, num_levels):
            cur = inputs[i]
            down = F.max_pool2d(bu_feats[i - 1], kernel_size=3, stride=2, padding=1)
            if down.shape[-2:] != cur.shape[-2:]:
                down = F.interpolate(down, size=cur.shape[-2:], mode='bilinear', align_corners=False)
            w = self._normalize_weights(self.w_bu[i], self.eps)
            fused = self._swish(w[0] * cur + w[1] * td_feats[i] + w[2] * down)
            bu_feats[i] = self.bu_convs[i](fused)

        return bu_feats
