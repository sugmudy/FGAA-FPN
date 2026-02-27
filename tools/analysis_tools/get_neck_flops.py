# Copyright (c) OpenMMLab. All rights reserved.
#
# Compute Params/FLOPs for the neck module only.
#
# Example:
#   python tools/analysis_tools/get_neck_flops.py configs/oriented_rcnn/BVAM.py --shape 1024 1024
#
import argparse
import math

import numpy as np
import torch
from mmcv import Config, DictAction

from mmrotate.models import build_neck

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError as e:
    raise ImportError('Please upgrade mmcv to >0.6.2') from e


class NeckWrapper(torch.nn.Module):
    """A wrapper that takes an image tensor and feeds dummy features to neck.

    This avoids counting backbone/head FLOPs and keeps the interface compatible
    with mmcv.get_model_complexity_info which expects a single tensor input.
    """

    def __init__(self, neck, in_channels, base_stride=4):
        super().__init__()
        self.neck = neck
        self.in_channels = list(in_channels)
        self.base_stride = int(base_stride)

    def forward(self, img):
        # img: (B, 3, H, W)
        b, _, h, w = img.shape
        feats = []
        for i, c in enumerate(self.in_channels):
            stride = self.base_stride * (2**i)
            fh = int(math.ceil(h / stride))
            fw = int(math.ceil(w / stride))
            feats.append(img.new_zeros((b, int(c), fh, fw)))
        return self.neck(tuple(feats))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute Params/FLOPs for neck only')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1024, 1024],
        help='input image size (h w) or single number for square')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options, key=value pairs')
    parser.add_argument(
        '--size-divisor',
        type=int,
        default=32,
        help='pad input to be divisible by this, -1 disables padding')
    parser.add_argument(
        '--base-stride',
        type=int,
        default=4,
        help='stride of the first backbone feature (usually 4 for ResNet C2)')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='device used for flops computation')
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid --shape, use: --shape 1024 1024')

    ori_shape = (3, h, w)
    divisor = args.size_divisor
    if divisor and divisor > 0:
        h = int(np.ceil(h / divisor)) * divisor
        w = int(np.ceil(w / divisor)) * divisor
    input_shape = (3, h, w)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if 'model' not in cfg or 'neck' not in cfg.model:
        raise KeyError('config does not contain cfg.model.neck')

    neck = build_neck(cfg.model.neck)
    neck.eval()

    in_channels = cfg.model.neck.get('in_channels', None)
    if in_channels is None:
        raise KeyError('cfg.model.neck.in_channels is required for this tool')

    wrapper = NeckWrapper(neck, in_channels, base_stride=args.base_stride)
    wrapper.eval()

    device = torch.device(args.device)
    wrapper.to(device)

    # Params (authoritative)
    params = sum(p.numel() for p in wrapper.neck.parameters())

    flops, mmcv_params = get_model_complexity_info(
        wrapper, input_shape, as_strings=True, print_per_layer_stat=False)

    split_line = '=' * 30
    if divisor and divisor > 0 and input_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {input_shape}\n')

    print(
        f'{split_line}\n'
        f'Config: {args.config}\n'
        f'Neck: {cfg.model.neck.get("type", "<unknown>")}\n'
        f'Input shape: {input_shape}\n'
        f'FLOPs(neck): {flops}\n'
        f'Params(neck): {params:,} (mmcv: {mmcv_params})\n'
        f'{split_line}')
    print('Note: FLOPs is an estimate. Some ops may be unsupported by mmcv.')


if __name__ == '__main__':
    main()
