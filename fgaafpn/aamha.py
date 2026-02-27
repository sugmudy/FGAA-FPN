# Copyright (c) OpenMMLab. All rights reserved.
# AAMHA module.

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule


class AngleAwareMHABlock(BaseModule):

    def __init__(self,
                 channels: int,
                 num_heads: int = 4,
                 embed_dim: Optional[int] = None,
                 attn_dropout: float = 0.0,
                 beta: float = 1.0,
                 orient_scale: float = 1.0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_heads = num_heads
        self.embed_dim = embed_dim if embed_dim is not None else channels
        assert self.embed_dim % num_heads == 0, \
            'embed_dim must be divisible by num_heads'
        self.head_dim = self.embed_dim // num_heads
        self.beta = beta
        self.orient_scale = orient_scale

        self.q_proj = nn.Conv2d(channels, self.embed_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, self.embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, self.embed_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(self.embed_dim, channels, kernel_size=1)

        self.orient_vec = nn.Parameter(torch.randn(num_heads, 2))
        nn.init.normal_(self.orient_vec, mean=0.0, std=0.5)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.norm = nn.GroupNorm(num_groups=32 if channels >= 32 else 1,
                                 num_channels=channels)

    def _shape_proj(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        x = x.view(B, self.num_heads, self.head_dim, H, W)
        x = x.view(B, self.num_heads, self.head_dim, H * W)
        x = x.permute(0, 1, 3, 2)
        return x

    def _compute_orientation_bias(self,
                                  H: int,
                                  W: int,
                                  device: torch.device) -> torch.Tensor:

        ys = torch.arange(H, dtype=torch.float32, device=device)
        xs = torch.arange(W, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1)
        coords = coords.view(-1, 2)
        N = coords.size(0)

        delta = coords.unsqueeze(1) - coords.unsqueeze(0)
        r = torch.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2) + 1e-6
        u = delta / r.unsqueeze(-1)

        u = u.unsqueeze(0)
        w = self.orient_vec.view(self.num_heads, 1, 1, 2)
        orient = (u * w).sum(dim=-1)
        return orient

    def _compute_fg_bias(self,
                         M: Optional[torch.Tensor],
                         H: int,
                         W: int) -> Optional[torch.Tensor]:

        if M is None:
            return None
        B = M.size(0)
        if M.shape[-2:] != (H, W):
            M = F.interpolate(M, size=(H, W),
                              mode='bilinear', align_corners=False)
        M = M.view(B, -1)
        N = M.size(1)
        Mp = M.unsqueeze(2)
        Mq = M.unsqueeze(1)
        fg_bias = Mp * (2.0 * Mq - 1.0)
        fg_bias = fg_bias.unsqueeze(1)
        return fg_bias

    def forward_single(self,
                       x: torch.Tensor,
                       M: Optional[torch.Tensor] = None) -> torch.Tensor:

        B, C, H, W = x.shape
        device = x.device
        N = H * W

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._shape_proj(q)
        k = self._shape_proj(k)
        v = self._shape_proj(v)

        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(self.head_dim)

        orient_bias = self._compute_orientation_bias(H, W, device)
        attn_logits = attn_logits + self.orient_scale * orient_bias.unsqueeze(0)

        fg_bias = self._compute_fg_bias(M, H, W)
        if fg_bias is not None and self.beta != 0.0:
            attn_logits = attn_logits + self.beta * fg_bias

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(B, self.embed_dim, H, W)

        out = self.out_proj(out)
        out = x + out
        out = self.norm(out)
        return out

    def forward(self,
                feats: List[torch.Tensor],
                fg_masks: Optional[List[Optional[torch.Tensor]]] = None,
                attn_levels: Optional[List[int]] = None) -> List[torch.Tensor]:

        num_levels = len(feats)
        if attn_levels is None:
            attn_levels = list(range(num_levels))
        if fg_masks is None:
            fg_masks = [None] * num_levels

        outs = []
        for lvl in range(num_levels):
            x = feats[lvl]
            M = fg_masks[lvl] if lvl < len(fg_masks) else None
            if lvl in attn_levels:
                out = self.forward_single(x, M)
            else:
                out = x
            outs.append(out)
        return outs
