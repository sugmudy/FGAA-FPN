# Copyright (c) OpenMMLab. All rights reserved.
# Main FGAAFPN neck.

from typing import Dict, List, Optional
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models.builder import NECKS

from .aamha import AngleAwareMHABlock
from .bifpn import BiFPNBlock
from .fgfm import FGSegHead, FGWeightNet
from mmcv.cnn import ConvModule

def debug_print(*args, prob=0.004):
    if random.random() < prob:
        print(*args)


@NECKS.register_module()
class FGAAFPN(BaseModule):

    def __init__(
            self,
            in_channels: List[int],
            out_channels: int,
            num_outs: int,
            start_level: int = 0,
            end_level: int = -1,
            add_extra_convs: bool = False,
            relu_before_extra_convs: bool = False,
            no_norm_on_lateral: bool = False,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=dict(type='ReLU'),
            upsample_cfg=dict(mode='bilinear', align_corners=False),
            init_cfg=dict(
                type='Xavier', layer='Conv2d', distribution='uniform'),

            num_bifpn_layers: int = 2,
            use_separable_conv: bool = True,

            fgfm_enable: bool = False,
            fgfm_levels: Optional[List[int]] = None,
            fgfm_alpha: float = 0.5,
            fgfm_fg_loss_weight: float = 0.3,

            use_attn: bool = True,
            attn_levels: Optional[List[int]] = None,
            attn_num_heads: int = 4,
            attn_embed_dim: Optional[int] = None,
            attn_dropout: float = 0.0,
            attn_beta: float = 1.0,
            attn_orient_scale: float = 1.0,

    ):
        super().__init__(init_cfg=init_cfg)

        self._debug_vis_counter = 0

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        self.fp16_enabled = False

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=None,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_fpn_convs = nn.ModuleList()
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_c = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_c = out_channels
                extra_fpn_conv = ConvModule(
                    in_c,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.extra_fpn_convs.append(extra_fpn_conv)
        elif extra_levels > 0:
            for _ in range(extra_levels):
                self.extra_fpn_convs.append(None)

        self.num_bifpn_layers = int(num_bifpn_layers)
        self.bifpn_blocks = nn.ModuleList()
        for _ in range(self.num_bifpn_layers):
            self.bifpn_blocks.append(
                BiFPNBlock(
                    in_channels=out_channels,
                    num_outs=num_outs,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    use_separable_conv=use_separable_conv))

        self.fgam_enable = fgfm_enable
        self.fgam_alpha = float(fgfm_alpha)
        self.fgam_fg_loss_weight = float(fgfm_fg_loss_weight)

        if fgfm_levels is None:
            self.fgam_levels = list(range(num_outs))
        else:
            self.fgam_levels = sorted(
                [int(lvl) for lvl in fgfm_levels if 0 <= lvl < num_outs])

        self.fg_heads = nn.ModuleList()
        if self.fgam_enable:
            for lvl in range(num_outs):
                if lvl in self.fgam_levels:
                    self.fg_heads.append(
                        FGSegHead(
                            in_channels=out_channels,
                            mid_channels=64,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg))
                else:
                    self.fg_heads.append(nn.Identity())
        else:
            self.fg_heads = nn.ModuleList([nn.Identity() for _ in range(num_outs)])

        self.fg_weight_nets = nn.ModuleList()
        if self.fgam_enable:
            for lvl in range(num_outs):
                if lvl in self.fgam_levels:
                    self.fg_weight_nets.append(FGWeightNet(out_channels))
                else:
                    self.fg_weight_nets.append(nn.Identity())
        else:
            self.fg_weight_nets = nn.ModuleList([nn.Identity() for _ in range(num_outs)])

        self.use_attn = use_attn
        if attn_levels is None:
            self.attn_levels = list(range(num_outs))
        else:
            self.attn_levels = sorted(
                [int(lvl) for lvl in attn_levels if 0 <= lvl < num_outs])

        if self.use_attn:
            self.attn_block = AngleAwareMHABlock(
                channels=out_channels,
                num_heads=attn_num_heads,
                embed_dim=attn_embed_dim if attn_embed_dim is not None else out_channels,
                attn_dropout=attn_dropout,
                beta=attn_beta,
                orient_scale=attn_orient_scale)
        else:
            self.attn_block = None

        self._cached_fg_masks: Optional[List[Optional[torch.Tensor]]] = None

    def forward(self, inputs: List[torch.Tensor]) -> tuple:

        assert len(inputs) == self.num_ins

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        fg_masks: List[Optional[torch.Tensor]] = [None] * self.num_outs

        if self.fgam_enable:
            enhanced_laterals: List[torch.Tensor] = []
            num_lateral_levels = len(laterals)

            for lvl in range(num_lateral_levels):
                x = laterals[lvl]
                head = self.fg_heads[lvl]

                if isinstance(head, FGSegHead) and (lvl in self.fgam_levels):
                    M_for_loss = head(x.detach())
                    M_for_loss = torch.nan_to_num(M_for_loss, nan=0.0, posinf=1.0, neginf=0.0)

                    fg_masks[lvl] = M_for_loss

                    M_for_gate = M_for_loss.detach()
                    M_prime = self.fg_weight_nets[lvl](x, M_for_gate)
                    M_prime = torch.nan_to_num(M_prime, nan=0.0, posinf=1.0, neginf=0.0)

                    x = x * (1.0 + self.fgam_alpha * M_prime)

                    debug_print(f"[FG] lvl={lvl} M:",
                                M_for_loss.mean().item(), M_for_loss.std().item(),
                                M_for_loss.min().item(), M_for_loss.max().item(), prob=0.005)
                    debug_print(f"[FG] lvl={lvl} M':",
                                M_prime.mean().item(), M_prime.std().item(),
                                M_prime.min().item(), M_prime.max().item(), prob=0.005)

                enhanced_laterals.append(x)

            laterals = enhanced_laterals
        else:
            fg_masks = [None] * self.num_outs

        self._cached_fg_masks = fg_masks

        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)

        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        if len(self.extra_fpn_convs) > 0:
            last_feat = outs[-1]
            for extra_conv in self.extra_fpn_convs:
                if extra_conv is not None:
                    last_feat = extra_conv(last_feat)
                else:
                    last_feat = F.max_pool2d(last_feat, kernel_size=1, stride=2)
                outs.append(last_feat)

        feats = outs

        from mmcv.runner import get_dist_info

        def _tensor_fingerprint(x: torch.Tensor):
            return {
                "mean": x.mean().item(),
                "std": x.std().item(),
                "absmax": x.abs().max().item(),
                "sum": x.sum().item(),
                "l2": x.norm().item(),
                "finite": bool(torch.isfinite(x).all().item()),
            }

        if self.use_attn and self.attn_block is not None:
            rank, _ = get_dist_info()

            do_dbg = (random.random() < 0.003)
            if do_dbg:
                before = {}
                with torch.no_grad():
                    for lvl in self.attn_levels:
                        if 0 <= lvl < len(feats):
                            before[lvl] = feats[lvl].detach().clone()

            feats = self.attn_block(
                feats=feats,
                fg_masks=fg_masks,
                attn_levels=self.attn_levels
            )

            if do_dbg and rank == 0:
                with torch.no_grad():
                    print("\n[DBG][ATTN] ===== compare feats before/after attn_block =====")
                    for lvl in self.attn_levels:
                        if lvl not in before or lvl >= len(feats):
                            print(f"[DBG][ATTN] lvl={lvl} skipped (no before or out of range)")
                            continue
                        a = before[lvl]
                        b = feats[lvl].detach()

                        diff = (b - a)
                        exactly_equal = torch.equal(a, b)
                        allclose = torch.allclose(a, b, rtol=1e-5, atol=1e-6)

                        denom = a.norm() + 1e-6
                        ratio = diff.norm() / denom

                        fa = _tensor_fingerprint(a)
                        fb = _tensor_fingerprint(b)

                        print(
                            f"[DBG][ATTN] lvl={lvl} shape={tuple(a.shape)} "
                            f"equal={exactly_equal} allclose={allclose} "
                            f"max|delta|={diff.abs().max().item():.3e} mean|delta|={diff.abs().mean().item():.3e} "
                            f"||delta||/||x||={ratio.item():.3e} "
                            f"sum: {fa['sum']:.3e}->{fb['sum']:.3e} "
                            f"mean: {fa['mean']:.3e}->{fb['mean']:.3e} "
                            f"std: {fa['std']:.3e}->{fb['std']:.3e} "
                            f"finite={fa['finite'] and fb['finite']}"
                        )
                    print("[DBG][ATTN] ================================================\n")

        for bifpn in self.bifpn_blocks:
            feats = bifpn(feats)

        return tuple(feats)

    def get_fgfm_loss(self,
                      img_metas: List[Dict],
                      gt_bboxes: List[torch.Tensor]) -> Dict[str, torch.Tensor]:

        if (not self.training) or (not self.fgam_enable):
            return dict()
        if self.fgam_fg_loss_weight <= 0.0:
            return dict()
        if self._cached_fg_masks is None:
            return dict()

        device = None
        for M in self._cached_fg_masks:
            if M is not None:
                device = M.device
                break

        if device is None:
            return dict()

        total_fg_loss = torch.zeros(1, device=device)
        fg_levels = 0

        for lvl, M_pred in enumerate(self._cached_fg_masks):
            if M_pred is None:
                continue
            if lvl not in self.fgam_levels:
                continue

            B, _, H, W = M_pred.shape
            M_pred_prob = M_pred.squeeze(1)

            if B > 0 and lvl == 0:
                if random.random() < 0.05:

                    self._debug_vis_counter += 1
                    self._debug_vis_fg_mask_single(
                        M_single=M_pred_prob[0],
                        img_meta=img_metas[0],
                        gt_bboxes_single=gt_bboxes[0],
                        lvl=lvl,
                        thr=0.7,
                        save_dir='work_dirs2/fg_vis2',
                        step_idx=self._debug_vis_counter)

            M_gt = M_pred_prob.new_zeros((B, H, W))
            has_pos_any = False

            for b in range(B):
                if len(gt_bboxes[b]) == 0:
                    continue

                if 'pad_shape' in img_metas[b]:
                    img_h, img_w = img_metas[b]['pad_shape'][:2]
                else:
                    img_h, img_w = img_metas[b]['img_shape'][:2]

                stride_y = img_h / float(H)
                stride_x = img_w / float(W)

                gt_b = gt_bboxes[b]
                if not isinstance(gt_b, torch.Tensor):
                    gt = gt_b.tensor
                else:
                    gt = gt_b
                gt = gt.detach().cpu().numpy()

                M_np = np.zeros((H, W), dtype=np.float32)
                has_pos = False

                for k in range(gt.shape[0]):
                    cx, cy, bw, bh, theta = gt[k]

                    theta_rad = float(theta)

                    dx = bw / 2.0
                    dy = bh / 2.0

                    corners = np.array([
                        [-dx, -dy],
                        [dx, -dy],
                        [dx, dy],
                        [-dx, dy],
                    ], dtype=np.float32)

                    cos_t = math.cos(theta_rad)
                    sin_t = math.sin(theta_rad)
                    R = np.array([
                        [cos_t, -sin_t],
                        [sin_t, cos_t],
                    ], dtype=np.float32)

                    rot = corners @ R.T
                    rot[:, 0] += cx
                    rot[:, 1] += cy

                    rot[:, 0] /= stride_x
                    rot[:, 1] /= stride_y

                    poly = np.round(rot).astype(np.int32)
                    poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
                    poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)

                    if cv2.contourArea(poly) <= 0:
                        continue

                    has_pos = True
                    cv2.fillConvexPoly(M_np, poly, 1.0)

                if has_pos:
                    has_pos_any = True
                    M_gt[b] = torch.from_numpy(M_np).to(device)

            if not has_pos_any:
                continue

            pos_mask = (M_gt == 1.0).float()
            neg_mask = (M_gt == 0.0).float()

            num_pos = pos_mask.sum()
            num_neg = neg_mask.sum()

            if num_pos > 0:
                pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=20.0)
            else:
                pos_weight = M_gt.new_tensor(1.0)

            pixel_weight = neg_mask + pos_weight * pos_mask

            bce_loss = F.binary_cross_entropy(
                input=M_pred_prob,
                target=M_gt,
                weight=pixel_weight,
                reduction='mean')

            eps = 1e-6
            p = M_pred_prob.view(B, -1)
            g = M_gt.view(B, -1)

            intersection = (p * g).sum(dim=1)
            union = p.sum(dim=1) + g.sum(dim=1)
            dice_loss = 1.0 - (2.0 * intersection + eps) / (union + eps)
            dice_loss = dice_loss.mean()

            lambda_dice = 0.6
            fg_loss = bce_loss + lambda_dice * dice_loss

            total_fg_loss = total_fg_loss + fg_loss
            fg_levels += 1

        if fg_levels == 0:
            return dict()

        loss_fgam_fg = total_fg_loss / fg_levels * self.fgam_fg_loss_weight
        return dict(loss_fgam_fg=loss_fgam_fg)

    def _debug_vis_fg_mask_single(self,
                                  M_single: torch.Tensor,
                                  img_meta: dict,
                                  gt_bboxes_single,
                                  lvl: int,
                                  thr: float = 0.7,
                                  save_dir: str = 'work_dirs/fg_vis1',
                                  epoch: int = -1,
                                  step_idx: int = -1):

        import os
        os.makedirs(save_dir, exist_ok=True)

        img_path = img_meta['filename']
        img = cv2.imread(img_path)
        if img is None:
            return

        if 'img_shape' in img_meta:
            h_new, w_new = img_meta['img_shape'][:2]
            img = cv2.resize(img, (w_new, h_new))
        img_h, img_w = img.shape[:2]

        flip_flag = img_meta.get('flip', False)
        if flip_flag:
            flip_dir = img_meta.get('flip_direction', 'horizontal')
            if flip_dir == 'horizontal':
                img = cv2.flip(img, 1)
            elif flip_dir == 'vertical':
                img = cv2.flip(img, 0)
            elif flip_dir == 'diagonal':
                img = cv2.flip(img, -1)

        angle = 0.0
        if 'rotated_angle' in img_meta:
            angle = img_meta['rotated_angle']
        elif 'rotate_angle' in img_meta:
            angle = img_meta['rotate_angle']
        if abs(angle) > 1e-3:
            center = (img_w * 0.5, img_h * 0.5)
            M_affine = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(
                img, M_affine, (img_w, img_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

        vis = img.copy()

        if not isinstance(gt_bboxes_single, torch.Tensor):
            gt_np = gt_bboxes_single.tensor.detach().cpu().numpy()
        else:
            gt_np = gt_bboxes_single.detach().cpu().numpy()

        for box in gt_np:
            cx, cy, w, h, theta = box
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            dx = w / 2.0
            dy = h / 2.0
            corners = np.array([
                [-dx, -dy],
                [dx, -dy],
                [dx, dy],
                [-dx, dy],
            ], dtype=np.float32)

            R = np.array([[cos_t, -sin_t],
                          [sin_t, cos_t]], dtype=np.float32)
            rot = corners @ R.T
            rot[:, 0] += cx
            rot[:, 1] += cy

            rot = rot.astype(np.int32)
            cv2.polylines(vis, [rot], isClosed=True,
                          color=(0, 255, 0), thickness=2)

        M_np = M_single.detach().cpu().numpy()
        M_up = cv2.resize(M_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        mask = (M_up > thr).astype(np.uint8)

        red = np.zeros_like(vis)
        red[:, :, 2] = 255

        alpha = 0.5
        mask3 = mask[:, :, None]
        vis = np.where(
            mask3 == 1,
            (alpha * red + (1.0 - alpha) * vis).astype(np.uint8),
            vis
        )

        step_str = f"s{step_idx:06d}" if step_idx >= 0 else "s_unk"
        epoch_str = f"e{epoch:03d}" if epoch >= 0 else "e_unk"
        base = os.path.basename(img_path)
        stem = os.path.splitext(base)[0]
        out_path = os.path.join(save_dir, f'{step_str}_lvl{lvl}_thr{thr:.2f}.jpg')
        cv2.imwrite(out_path, vis)


@NECKS.register_module()
class ABMAFPN(FGAAFPN):
    pass
