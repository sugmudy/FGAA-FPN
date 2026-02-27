# FGAA-FPN
Foreground-Guided Angle-Aware Network for Enhanced Oriented Object Detection
# FGAA-FPN

Foreground-Guided Angle-Aware Feature Pyramid Network for Rotated Object Detection (based on MMRotate).

## 1. 项目简介
FGAA-FPN 是一个面向旋转目标检测的 Neck 结构，在传统 FPN 的基础上引入三个核心模块：

- Foreground-Guided Feature Modulation (FGFM): 通过前景掩码对金字塔特征进行门控增强。
- Angle-Aware Multi-Head Attention (AAMHA): 在选定尺度上施加带方向偏置和前景偏置的自注意力。
- BiFPN Refinement: 通过可学习融合权重进行双向多尺度特征融合。

该实现已接入当前仓库的两阶段检测器训练流程：
`RotatedTwoStageDetector.forward_train` 会自动额外收集 `neck.get_fgam_loss(...)` 返回的损失。

## 2. 方法结构
整体前向流程：

1. Backbone 输出多尺度特征 `C2~C5`。
2. FPN 侧向卷积 + 自顶向下融合，得到初始 `P` 特征。
3. FGFM（可选）在指定层预测前景图并进行门控调制。
4. AAMHA（可选）在指定层施加角度感知注意力。
5. BiFPN（可选）进行若干层双向融合。
6. 输出多尺度特征给 RPN/ROI Head。

## 3. 代码位置
核心代码位于：

- `mmrotate/models/necks/fgaafpn/FGAAFPN.py`
- `mmrotate/models/necks/fgaafpn/fgfm.py`
- `mmrotate/models/necks/fgaafpn/aamha.py`
- `mmrotate/models/necks/fgaafpn/bifpn.py`
- `mmrotate/models/necks/FGAA_FPN.py`（兼容导入入口）

示例配置：

- `configs/oriented_rcnn/FGAM.py`
- `configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90_test.py`

## 4. 环境安装
建议使用与当前仓库依赖一致的环境（`mmcv-full 1.x + mmdet 2.x`）。

```bash
conda create -n fgaafpn python=3.8 -y
conda activate fgaafpn

pip install -U openmim
mim install "mmcv-full>=1.5.0"
mim install "mmdet>=2.25.1,<3.0.0"

pip install -r requirements.txt
pip install -v -e .
```

说明：

- 若你已在现有 MMRotate 环境中开发，可跳过重复安装。
- CUDA / PyTorch / mmcv-full 版本需相互匹配。

## 5. 数据准备（以 DOTA 为例）
默认配置使用：`data/split_DOTA/`

目录示例：

```text
data/split_DOTA/
  train/
    images/
    labelTxt/
  val/
    images/
    labelTxt/
```

## 6. 快速开始
### 6.1 训练
单卡训练：

```bash
python tools/train.py configs/oriented_rcnn/FGAM.py
```

或使用你的测试配置：

```bash
python tools/train.py configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90_test.py
```

多卡训练：

```bash
bash tools/dist_train.sh configs/oriented_rcnn/FGAM.py 8
```

### 6.2 测试与评估
```bash
python tools/test.py \
  configs/oriented_rcnn/FGAM.py \
  work_dirs/xxx/latest.pth \
  --eval mAP
```

多卡测试：

```bash
bash tools/dist_test.sh \
  configs/oriented_rcnn/FGAM.py \
  work_dirs/xxx/latest.pth \
  8 --eval mAP
```

## 7. 配置方式
你可以通过 `custom_imports` 加载 FGAA-FPN。

推荐写法（兼容入口）：

```python
custom_imports = dict(
    imports=['mmrotate.models.necks.FGAA_FPN'],
    allow_failed_imports=False
)

model = dict(
    type='OrientedRCNN',
    neck=dict(
        type='FGAAFPN',  # 或 ABMAFPN（ABMAFPN 是 FGAAFPN 的别名子类）
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        num_bifpn_layers=2,
        use_separable_conv=True,
        fgam_enable=True,
        fgam_levels=[2, 3, 4],
        fgam_alpha=0.8,
        fgam_fg_loss_weight=0.7,
        use_attn=True,
        attn_levels=[2, 3, 4],
        attn_num_heads=4,
        attn_embed_dim=256,
        attn_dropout=0.1,
        attn_beta=0.6,
        attn_orient_scale=0.7,
    )
)
```

## 8. 参数说明（FGAAFPN）
以下参数定义于 `FGAAFPN.__init__`。

### 8.1 FPN 基础参数
| 参数 | 默认值 | 说明 | 建议 |
|---|---:|---|---|
| `in_channels` | 必填 | Backbone 各层通道数列表 | ResNet50 常用 `[256,512,1024,2048]` |
| `out_channels` | 必填 | FPN 输出通道数 | 常用 `256` |
| `num_outs` | 必填 | 输出特征层数 | 常用 `5` |
| `start_level` | `0` | 从第几个 backbone stage 开始构建金字塔 | 常用 `0` |
| `end_level` | `-1` | 结束层，`-1` 表示用到最后一层 | 一般保持 `-1` |
| `add_extra_convs` | `False` | 是否用卷积生成额外层；否则用 max-pool 下采样 | 初期建议 `False` |
| `relu_before_extra_convs` | `False` | 额外卷积前是否加 ReLU | 与上项联动 |
| `no_norm_on_lateral` | `False` | lateral 1x1 conv 是否去掉 norm | 数据少时可尝试 `True` |
| `conv_cfg` | `None` | 卷积配置（MMCV 风格） | 通常保持 `None` |
| `norm_cfg` | `None` | 归一化配置 | 常用 `dict(type='BN', requires_grad=True)` |
| `act_cfg` | `dict(type='ReLU')` | 激活函数配置 | 常用默认 |
| `upsample_cfg` | `dict(mode='bilinear', align_corners=False)` | FPN 上采样配置 | 默认即可 |
| `init_cfg` | Xavier uniform | 初始化配置 | 默认即可 |

### 8.2 BiFPN 参数
| 参数 | 默认值 | 说明 | 建议 |
|---|---:|---|---|
| `num_bifpn_layers` | `2` | BiFPN 堆叠层数；`0` 表示关闭 | 先从 `0/1/2` 网格搜索 |
| `use_separable_conv` | `True` | BiFPN 内是否用深度可分离卷积 | 显存紧张时建议 `True` |

### 8.3 前景调制（FGFM）参数
| 参数 | 默认值 | 说明 | 建议 |
|---|---:|---|---|
| `fgam_enable` | `False` | 是否启用前景分支与门控调制 | 使用 FGAA-FPN 时设为 `True` |
| `fgam_levels` | `None` | 启用 FGFM 的层索引列表（相对输出金字塔） | 常用 `[2,3,4]` 或 `[0,1,2]` |
| `fgam_alpha` | `0.5` | 门控强度，作用在 `x * (1 + alpha * M')` | 常用 `0.5~1.0` |
| `fgam_fg_loss_weight` | `0.3` | 前景监督损失总权重 | 从 `0.1~1.0` 调参 |

补充：

- `fgam_levels=None` 时默认对全部输出层生效。
- 前景损失 `loss_fgam_fg` 只在 `training=True`、`fgam_enable=True`、`fgam_fg_loss_weight>0` 时返回。

### 8.4 角度注意力（AAMHA）参数
| 参数 | 默认值 | 说明 | 建议 |
|---|---:|---|---|
| `use_attn` | `True` | 是否启用角度感知多头注意力 | 初期可先关掉做消融 |
| `attn_levels` | `None` | 启用注意力的层索引列表 | 常用 `[2,3,4]` |
| `attn_num_heads` | `4` | 注意力头数 | `4` 或 `8` |
| `attn_embed_dim` | `None` | 注意力嵌入维度；`None` 等于 `out_channels` | 常用与 `out_channels` 相同 |
| `attn_dropout` | `0.0` | 注意力 dropout | 小数据集可设 `0.1` |
| `attn_beta` | `1.0` | 前景偏置强度 | 常用 `0.3~1.0` |
| `attn_orient_scale` | `1.0` | 方向偏置强度 | 常用 `0.3~1.0` |

## 9. 损失与训练行为
`FGAAFPN.get_fgam_loss` 当前实现：

- 使用旋转框 rasterize 到各层特征图，生成弱监督前景 GT。
- 损失形式：`BCE(带前景重加权) + 0.6 * Dice`。
- 多层损失求平均后乘以 `fgam_fg_loss_weight`，键名为 `loss_fgam_fg`。

总损失（两阶段检测器）可写作：

```text
loss_total = loss_rpn + loss_roi + loss_fgam_fg(若启用)
```

## 10. 可视化与调试
FGAA-FPN 内置了低频率调试输出：

- 随机打印前景统计与注意力前后差异（用于排查梯度/数值问题）。
- 随机保存前景掩码可视化到 `work_dirs2/fg_vis2/`。

如果你不希望训练期间产生调试输出，可将相关 `debug_print` 或可视化逻辑关闭。

## 11. 常见问题
### 11.1 `NameError: ConvModule is not defined`
原因：`FGAAFPN.py` 未导入 `ConvModule`。

修复：确保存在以下导入。

```python
from mmcv.cnn import ConvModule
```

### 11.2 配置中找不到 `FGAAFPN`
检查 `custom_imports` 是否正确，推荐：

```python
custom_imports = dict(
    imports=['mmrotate.models.necks.FGAA_FPN'],
    allow_failed_imports=False
)
```

### 11.3 训练显存不足
可依次尝试：

- 降低 `attn_levels` 数量。
- 关闭 `use_attn`。
- 减少 `num_bifpn_layers`。
- 开启 `use_separable_conv`。
- 降低输入分辨率或 batch size。

## 12. 结果复现建议
建议在论文/实验中固定并公开：

- 随机种子、GPU 型号、batch size、训练轮次。
- `fgam_levels / attn_levels / num_bifpn_layers` 三组关键超参。
- 使用的配置文件与 checkpoint。

## 13. 引用
如果你在研究中使用了 FGAA-FPN，请引用你的论文（将下面条目替换为你的正式信息）：

```bibtex
@article{your_fgaafpn_2026,
  title   = {FGAA-FPN: Foreground-Guided Angle-Aware Feature Pyramid Network for Rotated Object Detection},
  author  = {Your Name and Coauthors},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2026}
}
```

## 14. License
本项目代码建议使用 `Apache-2.0`（与 MMRotate 主体协议一致）。
