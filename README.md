# FGAA-FPN with Oriented R-CNN (MMRotate)

This README is aligned with the current implementation in:

- `configs/oriented_rcnn/oriented_rcnn_r50_FGAAFPN.py`
- `mmrotate/models/necks/fgaafpn/FGAAFPN.py`
- `mmrotate/models/necks/fgaafpn/fgfm.py`
- `mmrotate/models/necks/fgaafpn/bifpn.py`
- `mmrotate/models/necks/fgaafpn/aamha.py`
- `mmrotate/models/necks/fgaafpn/fg_vis.py`

## 1. Overview

FGAA-FPN is a custom neck for oriented object detection that extends FPN with:

- FGFM: Foreground-Guided Feature Modulation.
- AAMHA: Angle-Aware Multi-Head Attention.
- BiFPN: Learnable bi-directional multi-scale fusion.
- Optional foreground-mask visualization for debugging.

It is used in an Oriented R-CNN training setup on DOTA-style data.

## 2. What Each File Does

```text
mmrotate/models/necks/fgaafpn/
  FGAAFPN.py    -> Main neck: FPN + FGFM + AAMHA + BiFPN + FGFM loss
  fgfm.py       -> FGSegHead and FGWeightNet
  aamha.py      -> AngleAwareMHABlock
  bifpn.py      -> BiFPNBlock and DepthwiseSeparableConv
  fg_vis.py     -> _debug_vis_fg_mask_single visualization utility
  __init__.py   -> Exports modules
```

## 3. Forward Flow in FGAAFPN

1. Build lateral features and FPN conv outputs.
2. If `fgfm_enable=True`, compute foreground masks and gate features.
3. Perform standard top-down fusion.
4. Add extra levels if configured.
5. If `use_attn=True`, apply angle-aware attention on `attn_levels`.
6. Apply `num_bifpn_layers` BiFPN blocks.
7. Return tuple of multi-level feature maps.

## 4. FGFM Loss

`FGAAFPN.get_fgfm_loss(img_metas, gt_bboxes)`:

- Rasterizes rotated GT boxes to weak foreground masks per level.
- Computes weighted BCE + Dice (`lambda_dice=0.6`).
- Averages across selected levels.
- Returns `dict(loss_fgam_fg=...)`.

## 5. Foreground Visualization (Manual On/Off)

Visualization code is split from `FGAAFPN.py` into:

- `mmrotate/models/necks/fgaafpn/fg_vis.py`
- function `_debug_vis_fg_mask_single(...)`

Visualization is triggered only when:

- `fg_vis_enable=True`
- level is included in `fg_vis_levels`
- random sampling condition `rand < fg_vis_prob` is satisfied

Output path is controlled by `fg_vis_save_dir`.

## 6. Full Config Breakdown (`oriented_rcnn_r50_FGAAFPN.py`)

## 6.1 Global Config Keys

| Key | Role |
|---|---|
| `_base_` | Inherits dataset/schedule/runtime defaults |
| `metainfo` | Class names |
| `custom_imports` | Imports custom FGAA-FPN module before model build |
| `angle_version` | Angle convention (`le90`) |
| `model` | Full detector structure |
| `img_norm_cfg` | Image normalization mean/std |
| `train_pipeline` | Data preprocessing and augmentation |
| `data` | Dataset roots and split definitions |
| `work_dir` | Training output folder |
| `optimizer` | Optimizer hyperparameters |

## 6.2 `custom_imports`

Use:

```python
custom_imports = dict(
    imports=['mmrotate.models.necks.fgaafpn.FGAAFPN'],
    allow_failed_imports=False
)
```

This ensures `FGAAFPN` is registered before `build_detector`.

## 6.3 `model` Block

### Backbone (`ResNet`)

The current config uses standard ResNet-50 settings for Oriented R-CNN:

- `depth=50`
- `out_indices=(0,1,2,3)`
- `frozen_stages=1`
- BN normalization with `requires_grad=True`
- torchvision pretrained initialization

### Neck (`FGAAFPN`)

Current style:

```python
neck=dict(
    type='FGAAFPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5,
    start_level=0,
    end_level=-1,
    add_extra_convs=False,
    relu_before_extra_convs=False,

    num_bifpn_layers=0,
    use_separable_conv=False,

    fgfm_enable=True,
    fgfm_levels=[2,3,4],
    fgfm_alpha=0.8,
    fgfm_fg_loss_weight=0.7,

    fg_vis_enable=False,
    fg_vis_levels=[0],
    fg_vis_prob=0.05,
    fg_vis_thr=0.7,
    fg_vis_save_dir='work_dirs2/fg_vis2',

    use_attn=False,
    attn_levels=[2,3,4],
    attn_num_heads=4,
    attn_embed_dim=256,
    attn_dropout=0.1,
    attn_beta=0.6,
    attn_orient_scale=0.7,
)
```

### RPN Head (`OrientedRPNHead`)

Current config uses:

- AnchorGenerator with strides `[4,8,16,32,64]`
- MidpointOffsetCoder with `angle_range=le90`
- BCE classification loss + SmoothL1 bbox loss

### RoI Head (`OrientedStandardRoIHead`)

Current config uses:

- `RotatedSingleRoIExtractor` with `RoIAlignRotated`
- `RotatedShared2FCBBoxHead`
- `DeltaXYWHAOBBoxCoder` for final rotated box regression

### Train/Test Config

- RPN assigner/sampler and proposal settings are explicitly configured.
- RCNN assigner uses rotated IoU calculator (`RBboxOverlaps2D`).
- Test stage uses score threshold, NMS, and max-per-image settings.

## 6.4 Data and Pipeline

### `train_pipeline`

Current steps:

- `LoadImageFromFile`
- `LoadAnnotations(with_bbox=True)`
- `RResize(img_scale=(1024,1024))`
- `RRandomFlip` with 3 flip directions
- `Normalize`
- `Pad(size_divisor=32)`
- `DefaultFormatBundle`
- `Collect(keys=['img','gt_bboxes','gt_labels'])`



## 7. FGAAFPN Parameter Reference

### 7.1 Base FPN Parameters

| Parameter | Type | Meaning |
|---|---|---|
| `in_channels` | list[int] | Backbone feature channels |
| `out_channels` | int | Unified feature width |
| `num_outs` | int | Number of output levels |
| `start_level` | int | Start backbone level |
| `end_level` | int | End backbone level (`-1` means all) |
| `add_extra_convs` | bool/str | Add extra conv levels |
| `relu_before_extra_convs` | bool | ReLU before extra convs |
| `no_norm_on_lateral` | bool | Disable lateral norm |
| `conv_cfg` | dict/None | MMCV conv config |
| `norm_cfg` | dict/None | MMCV norm config |
| `act_cfg` | dict | Activation config |
| `upsample_cfg` | dict | Upsampling behavior |
| `init_cfg` | dict | Initialization settings |

### 7.2 BiFPN Parameters

| Parameter | Type | Meaning |
|---|---|---|
| `num_bifpn_layers` | int | Number of BiFPN blocks (`0` to disable) |
| `use_separable_conv` | bool | Use depthwise separable conv in BiFPN |

### 7.3 FGFM Parameters

| Parameter | Type | Meaning |
|---|---|---|
| `fgfm_enable` | bool | Enable foreground modulation |
| `fgfm_levels` | list[int]/None | Levels to apply FGFM |
| `fgfm_alpha` | float | Modulation strength |
| `fgfm_fg_loss_weight` | float | FGFM loss weight |

### 7.4 Visualization Parameters

| Parameter | Type | Meaning |
|---|---|---|
| `fg_vis_enable` | bool | Enable/disable visualization |
| `fg_vis_levels` | list[int]/None | Levels allowed for visualization |
| `fg_vis_prob` | float | Sampling probability |
| `fg_vis_thr` | float | Mask threshold |
| `fg_vis_save_dir` | str | Save directory |

### 7.5 Attention Parameters

| Parameter | Type | Meaning |
|---|---|---|
| `use_attn` | bool | Enable angle-aware attention |
| `attn_levels` | list[int]/None | Levels to run attention |
| `attn_num_heads` | int | Number of heads |
| `attn_embed_dim` | int/None | Attention embedding dim |
| `attn_dropout` | float | Attention dropout |
| `attn_beta` | float | Foreground-bias weight |
| `attn_orient_scale` | float | Orientation-bias weight |

## 8. How to Write Config for Typical Experiments

### Baseline-like setup

- `fgfm_enable=False`
- `use_attn=False`
- `num_bifpn_layers=0`
- `fg_vis_enable=False`

### FGFM-only setup

- `fgfm_enable=True`
- `use_attn=False`
- `num_bifpn_layers=0`

### FGFM + AAMHA setup

- `fgfm_enable=True`
- `use_attn=True`
- `num_bifpn_layers=0`

### Full FGAA-FPN setup

- `fgfm_enable=True`
- `use_attn=True`
- `num_bifpn_layers>0`

## 9. Train and Evaluate

Single GPU:

```bash
python tools/train.py configs/oriented_rcnn/oriented_rcnn_r50_FGAAFPN.py
```


Evaluation:

```bash
python tools/test.py \
  configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90_test.py \
  work_dirs6/oriented_rcnn_dota15_r50_fgamfpn_1x/latest.pth \
  --eval mAP
```
