# UG-CASMamba

UG-CASMamba 是一个面向 2D 心脏 MRI 多类别分割的 U-Net/Mamba 混合模型。本仓库保留模型定义与随附的 ACDC 2D 数据，不包含训练循环、数据增强、损失函数、指标计算或可视化脚本。

## 模型概览

默认输入为 `B × 1 × H × W` 的灰度图像，默认输出为 `B × 4 × H × W` 的分割 logits。四个类别的约定为：

| 类别编号 | 含义 |
| --- | --- |
| 0 | Background（背景） |
| 1 | RV（右心室） |
| 2 | Myo（心肌） |
| 3 | LV（左心室） |

整体数据流如下：

```text
输入图像
  │
  ├─ ConvBlock ─ x0 (16 channels)
  ├─ DownBlock ─ x1 (32 channels)
  ├─ DownBlock ─ x2_raw (64 channels)
  │                  │
  │                  ├─ X2UncertaintyHead ─ 粗分割概率、熵不确定性、边界响应、gate2
  │                  ├─ UGM-Mamba(x2_raw, gate2)
  │                  └─ CASM(x2_raw, class probability) ─ 类别解剖关系图
  ├─ DownBlock ─ x3 (128 channels) ─ UGM-Mamba(gate2)
  ├─ DownBlock ─ x4 (256 channels) ─ UGM-Mamba(gate2)
  │
  └─ 4-stage U-Net decoder + skip connections ─ 4-class logits
```

对于 `256 × 256` 输入，默认编码器特征分辨率依次为 `256²、128²、64²、32²、16²`。

## 核心模块

### 1. X2 uncertainty head

模型在 `x2` 尺度生成一个粗分割结果，并从类别概率构造两个引导信号：

- 归一化类别熵，用于表示像素级预测不确定性；
- 心脏前景概率的空间梯度，用于表示候选边界。

二者经过卷积与 Sigmoid 得到 `gate2`。该 gate 会被缩放到 `x3`、`x4` 的空间尺寸，形成多尺度不确定性引导。

### 2. UGM-Mamba

`LightSS2D` 沿水平、垂直及其反向共四个方向执行 2D selective scan。UGM-Mamba 将不确定性 gate 用于：

1. 校准 selective scan 的输入相关步长 `delta/dt`；
2. 动态预测四个扫描方向的融合权重；
3. 调节 Mamba 输出与原始特征之间的残差更新强度。

UGM-Mamba 分别作用在 `x2`、`x3`、`x4`，用于在不同尺度上增强不确定区域和边界区域的上下文建模。

### 3. CASM

CASM（Class-wise Anatomical State Mamba）是位于 `x2` 的独立类别关系分支：

1. 使用粗分割概率对空间特征进行类别加权池化，得到 BG/RV/Myo/LV 四个类别 token；
2. 使用双向 `ClassMamba1D` 建模类别状态之间的解剖关系；
3. 按类别概率将更新后的 token 广播回空间，形成解剖关系图；
4. 通过可学习残差系数融合回 `x2` 特征。

### 4. Decoder

解码器采用四级双线性上采样和 U-Net skip connection，最终通过 `3 × 3` 卷积输出 `class_num` 个通道的分割 logits。

## 默认配置

| 参数 | 默认值 |
| --- | --- |
| `in_chns` | `1` |
| `class_num` | `4` |
| `feature_chns` | `(16, 32, 64, 128, 256)` |
| `dropout` | `(0.05, 0.1, 0.2, 0.3, 0.5)` |
| `d_state_x2/x3/x4` | `16/16/16` |
| `expand_x2/x3/x4` | `2.0/2.0/2.0` |
| `casm_d_state` | `8` |
| `casm_expand` | `2.0` |

## 安装

建议先根据本机 CUDA 环境安装匹配版本的 PyTorch，再安装其余依赖：

```bash
pip install -r requirements.txt
```

`mamba-ssm` 的 selective scan 通常需要可用的 CUDA/PyTorch 编译环境。若只想检查纯 U-Net 路径，可同时设置 `use_mamba=False` 和 `use_casm=False`。

## 使用示例

```python
import torch

from model import UNet_MultiScale_UGM_Mamba

model = UNet_MultiScale_UGM_Mamba(
    in_chns=1,
    class_num=4,
    use_mamba=True,
    use_casm=True,
)

x = torch.randn(2, 1, 256, 256)
logits = model(x)
prediction = logits.argmax(dim=1)

print(logits.shape)      # torch.Size([2, 4, 256, 256])
print(prediction.shape)  # torch.Size([2, 256, 256])
```

如需查看中间引导结果：

```python
logits, aux = model(x, return_aux=True)

print(aux.keys())
# coarse_logits_x2, prob_x2, uncertainty_x2, edge_x2, gate_x2,
# gate_x3, gate_x4, casm_x2, casm_map_x2
```

当 `use_casm=False` 时，CASM 相关的两个辅助输出不会出现。

## 数据目录

仓库中的数据来自随项目提供的 `ACDC-2D-All.zip`，解压后保持如下结构：

```text
data/ACDC-2D-All/
├─ train/
│  ├─ Img/                 # 1,394 张输入图像
│  ├─ GT/                  # 1,394 张完整标注
│  └─ WeaklyAnnotations/   # 1,394 张弱标注
└─ val/
   ├─ Img/                 # 404 张输入图像
   ├─ GT/                  # 404 张完整标注
   └─ WeaklyAnnotations/   # 404 张弱标注
```

图像、GT 和弱标注通过相同文件名一一对应。模型代码本身不包含数据读取或预处理逻辑，使用者可根据自己的训练流程构建 `Dataset` 和 `DataLoader`。

## 文件说明

- `model.py`：UG-CASMamba 的完整模型定义；
- `requirements.txt`：模型运行所需的最小 Python 依赖；
- `data/ACDC-2D-All/`：训练集与验证集的图像、完整标注和弱标注。
