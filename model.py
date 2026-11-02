"""UG-CASMamba model definitions extracted from the original training script.

This module contains model architecture code only. Training, preprocessing,
loss, metric, data-loading, and visualization code are intentionally omitted.
"""

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

try:
    from mamba_ssm import selective_scan_fn
except ImportError:
    selective_scan_fn = None

class ConvBlock(nn.Module):
    """Two convolution layers with BatchNorm and LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float):
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock."""
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling followed by ConvBlock."""
    def __init__(self, in_channels1: int, in_channels2: int, out_channels: int, dropout_p: float):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# =========================================================
# 3. Mamba components
# =========================================================

def trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.):
    """Truncated normal initialization."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class ChannelAttention(nn.Module):
    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        hidden = max(num_feat // squeeze_factor, 4)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_feat, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class CAB(nn.Module):
    """Convolutional attention block used after SS2D."""
    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 16):
        super().__init__()
        hidden = max(num_feat // compress_ratio, 4)
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, hidden, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, num_feat, kernel_size=3, stride=1, padding=1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


class LightSS2D(nn.Module):
    """
    Uncertainty-calibrated lightweight 2D selective scan block.

    Input : x    -> B, H, W, C
            gate -> B, 1, H, W or None
    Output: B, H, W, C

    Two uncertainty-aware designs are included:
    1) Uncertainty-calibrated selective scan:
       the uncertainty gate modulates the input-dependent delta/dts term
       before selective_scan_fn, so uncertain regions have stronger state update.
    2) Uncertainty-aware dynamic direction mixer:
       the four scan directions are fused using weights predicted from both
       feature and uncertainty gate.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank: Union[str, int] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()

        if selective_scan_fn is None:
            raise ImportError(
                "mamba_ssm is not installed. Install it first, e.g. `pip install mamba-ssm`."
            )

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Direction mixer now uses feature + uncertainty gate.
        hidden = max(self.d_inner // 8, 8)
        self.dir_mixer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.d_inner + 1, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 4, kernel_size=1),
        )

        # Learnable strength for delta/dts calibration.
        # Initialized to a small positive gain after sigmoid to avoid unstable early training.
        self.delta_gate_scale = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def _build_gate_sequences(self, gate: Optional[torch.Tensor], B: int, H: int, W: int, L: int, dtype, device):
        """
        Build gate sequences aligned with the four scan directions.
        Returns: B, 4, 1, L
        """
        if gate is None:
            gate = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        else:
            if gate.shape[-2:] != (H, W):
                gate = F.interpolate(gate, size=(H, W), mode="bilinear", align_corners=True)
            gate = gate.to(device=device, dtype=dtype).clamp(0.0, 1.0)

        gate_hwwh = torch.stack(
            [
                gate.view(B, 1, L),
                torch.transpose(gate, dim0=2, dim1=3).contiguous().view(B, 1, L),
            ],
            dim=1,
        ).view(B, 2, 1, L)
        gate_seq = torch.cat([gate_hwwh, torch.flip(gate_hwwh, dims=[-1])], dim=1)
        return gate, gate_seq

    def forward_core(self, x: torch.Tensor, gate: Optional[torch.Tensor] = None):
        """
        Args:
            x: B, C_inner, H, W
            gate: B, 1, H, W or None

        Returns:
            y1, y2, y3, y4: each B, C_inner, H*W
        """
        B, C, H, W = x.shape
        L = H * W
        K = 4

        _, gate_seq = self._build_gate_sequences(gate, B, H, W, L, x.dtype, x.device)

        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)

        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            xs.view(B, K, -1, L),
            self.x_proj_weight,
        )

        dts, Bs, Cs = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2,
        )

        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.view(B, K, -1, L),
            self.dt_projs_weight,
        )

        # Innovation 1: uncertainty-calibrated selective state update.
        # A small learnable factor scales dts in uncertain regions before selective_scan_fn.
        delta_gain = 0.5 * torch.sigmoid(self.delta_gate_scale)
        dts = dts * (1.0 + delta_gain * gate_seq)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)

        y1 = out_y[:, 0]
        y2 = inv_y[:, 0]
        y3 = torch.transpose(
            out_y[:, 1].view(B, -1, W, H),
            dim0=2,
            dim1=3,
        ).contiguous().view(B, -1, L)
        y4 = torch.transpose(
            inv_y[:, 1].view(B, -1, W, H),
            dim0=2,
            dim1=3,
        ).contiguous().view(B, -1, L)

        return y1, y2, y3, y4

    def forward(self, x: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: B, H, W, C
            gate: B, 1, H, W or None
        """
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        gate_map, _ = self._build_gate_sequences(gate, B, H, W, H * W, x.dtype, x.device)

        # Innovation 2: uncertainty-aware dynamic direction mixer.
        # The direction weights are predicted from feature + gate.
        dir_input = torch.cat([x, gate_map], dim=1)
        dir_weights = self.dir_mixer(dir_input).view(B, 4, 1, 1)
        dir_weights = torch.softmax(dir_weights, dim=1)

        y1, y2, y3, y4 = self.forward_core(x, gate=gate_map)
        ys = torch.stack([y1, y2, y3, y4], dim=1)       # B, 4, C_inner, L
        y = (ys * dir_weights).sum(dim=1)               # B, C_inner, L

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)

        return out


class VSSBlock2D(nn.Module):
    """
    Feature-map wrapper of uncertainty-calibrated LightSS2D.

    Input : feat -> B, C, H, W
            gate -> B, 1, H_g, W_g or None
    Output: B, C, H, W
    """
    def __init__(self, dim: int, d_state: int = 16, expand: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = LightSS2D(d_model=dim, d_state=d_state, expand=expand, dropout=dropout)
        self.skip_scale1 = nn.Parameter(torch.ones(dim))

        self.norm2 = nn.LayerNorm(dim)
        self.cab = CAB(dim)
        self.skip_scale2 = nn.Parameter(torch.ones(dim))

    def forward(self, feat: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = feat.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        y = self.ss2d(self.norm1(x), gate=gate)
        x = x * self.skip_scale1 + y

        z = self.norm2(x).permute(0, 3, 1, 2).contiguous()
        z = self.cab(z).permute(0, 2, 3, 1).contiguous()
        x = x * self.skip_scale2 + z

        return x.permute(0, 3, 1, 2).contiguous()


# =========================================================
# 4. x2 uncertainty head and external-gate UGM-Mamba
# =========================================================

class X2UncertaintyHead(nn.Module):
    """
    Generate coarse segmentation logits and uncertainty-boundary gate from x2.

    x2 feature: B, C, 64, 64 for 256x256 input.
    coarse_logits: B, num_classes, 64, 64
    gate2: B, 1, 64, 64
    """
    def __init__(self, in_channels: int, num_classes: int = 4, detach_prob_for_gate: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.detach_prob_for_gate = detach_prob_for_gate

        self.coarse_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

        self.gate_conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        coarse_logits = self.coarse_head(x2)
        prob = torch.softmax(coarse_logits, dim=1)

        prob_gate = prob.detach() if self.detach_prob_for_gate else prob

        eps = 1e-6
        # Entropy uncertainty, normalized to [0, 1]
        uncertainty = -torch.sum(prob_gate * torch.log(prob_gate + eps), dim=1, keepdim=True)
        uncertainty = uncertainty / math.log(self.num_classes)

        # class order: 0=BG, 1=RV, 2=Myo, 3=LV
        if self.num_classes >= 4:
            heart_prob = prob_gate[:, 1:4].sum(dim=1, keepdim=True)
        else:
            heart_prob = 1.0 - prob_gate[:, 0:1]

        # probability-gradient boundary response
        dx = torch.abs(heart_prob[:, :, :, 1:] - heart_prob[:, :, :, :-1])
        dy = torch.abs(heart_prob[:, :, 1:, :] - heart_prob[:, :, :-1, :])
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        edge = dx + dy
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)

        gate2 = self.gate_conv(torch.cat([uncertainty, edge], dim=1))

        aux = {
            "coarse_logits_x2": coarse_logits,
            "prob_x2": prob,
            "uncertainty_x2": uncertainty,
            "edge_x2": edge,
            "gate_x2": gate2,
        }
        return coarse_logits, gate2, aux


class ClassMamba1D(nn.Module):
    """
    Lightweight bidirectional 1D Mamba for class-state tokens.

    Input : B, K, C, where K is the number of classes, usually 4.
    Output: B, K, C.

    This module scans class tokens rather than spatial pixels. It is used by
    the independent CASM branch to model anatomical interactions among
    BG/RV/Myo/LV class states.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank: Union[str, int] = "auto",
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError(
                "mamba_ssm is not installed. Install it first, e.g. `pip install mamba-ssm`."
            )

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.num_dirs = 2

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=True,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            LightSS2D.dt_init(self.dt_rank, self.d_inner, 1.0, "random", 0.001, 0.1, 1e-4),
            LightSS2D.dt_init(self.dt_rank, self.d_inner, 1.0, "random", 0.001, 0.1, 1e-4),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = LightSS2D.A_log_init(self.d_state, self.d_inner, copies=self.num_dirs, merge=True)
        self.Ds = LightSS2D.D_init(self.d_inner, copies=self.num_dirs, merge=True)
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        hidden = max(self.d_inner // 8, 8)
        self.dir_mixer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.d_inner, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, self.num_dirs, kernel_size=1),
        )

    def forward_core(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: B, C_inner, K
        returns forward and backward class-state scan outputs.
        """
        B, C, L = x.shape
        K = self.num_dirs

        xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)  # B,2,C,L

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            xs.view(B, K, -1, L),
            self.x_proj_weight,
        )
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.view(B, K, -1, L),
            self.dt_projs_weight,
        )

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        y_forward = out[:, 0]
        y_backward = torch.flip(out[:, 1], dims=[-1])
        return y_forward, y_backward

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: B, K, C
        """
        B, K_len, C = tokens.shape
        xz = self.in_proj(tokens)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2).contiguous()  # B,C_inner,K
        x = self.act(self.conv1d(x))

        dir_weights = self.dir_mixer(x).view(B, self.num_dirs, 1, 1)
        dir_weights = torch.softmax(dir_weights, dim=1)

        y1, y2 = self.forward_core(x)
        ys = torch.stack([y1, y2], dim=1)   # B,2,C_inner,K
        y = (ys * dir_weights).sum(dim=1)   # B,C_inner,K

        y = y.transpose(1, 2).contiguous()  # B,K,C_inner
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class ClassWiseAnatomicalStateMamba(nn.Module):
    """
    Independent Class-wise Anatomical State Modeling branch, CASM.

    It is placed at x2 as an independent anatomical relationship branch:
        x2 feature + P2 probability map
            -> probability-weighted BG/RV/Myo/LV class tokens
            -> ClassMamba1D for class-state interaction
            -> probability-guided broadcasting
            -> anatomical relation map A2.

    It does not replace UC-Mamba. It provides a separate class-level anatomical
    relation map that is fused back to x2.
    """
    def __init__(
        self,
        channels: int,
        num_classes: int = 4,
        d_state: int = 8,
        expand: float = 2.0,
        detach_prob: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.detach_prob = detach_prob

        self.class_embed = nn.Parameter(torch.zeros(1, num_classes, channels))
        trunc_normal_(self.class_embed, std=0.02)

        self.class_mamba = ClassMamba1D(
            d_model=channels,
            d_state=d_state,
            expand=expand,
            dropout=0.0,
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, feat: torch.Tensor, prob: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        feat: B,C,H,W, usually x2_raw.
        prob: B,num_classes,H_p,W_p, usually prob_x2 from coarse head.
        """
        B, C, H, W = feat.shape
        if prob.shape[-2:] != feat.shape[-2:]:
            prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=True)

        prob_for_class = prob.detach() if self.detach_prob else prob
        prob_for_class = prob_for_class / (prob_for_class.sum(dim=1, keepdim=True) + 1e-6)

        # Probability-weighted class pooling: B,K,C
        prob_exp = prob_for_class.unsqueeze(2)  # B,K,1,H,W
        feat_exp = feat.unsqueeze(1)            # B,1,C,H,W
        denom = prob_exp.sum(dim=(3, 4)) + 1e-6 # B,K,1
        class_tokens = (feat_exp * prob_exp).sum(dim=(3, 4)) / denom

        class_tokens_with_embed = class_tokens + self.class_embed
        class_states = self.class_mamba(class_tokens_with_embed)  # B,K,C

        # Broadcast updated class states back to the spatial feature map.
        class_map = torch.einsum("bkc,bkhw->bchw", class_states, prob_for_class)
        class_map = self.out_conv(class_map)

        aux = {
            "class_tokens": class_tokens,
            "class_states": class_states,
            "class_prob": prob_for_class,
            "class_map": class_map,
        }
        return class_map, aux


class ExternalGateUGMMamba(nn.Module):
    """
    UGM-Mamba block with uncertainty-calibrated selective scan.

    Args:
        feat: B, C, H, W
        gate: B, 1, H_g, W_g. It will be resized to H, W.

    The same uncertainty gate is used in three places:
        1. Inside LightSS2D to calibrate dts/delta before selective_scan_fn.
        2. Inside LightSS2D to predict uncertainty-aware direction weights.
        3. Outside Mamba to modulate the residual feature update.
    """
    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        expand: float = 2.0,
        dropout: float = 0.0,
        init_gamma: float = 0.1,
    ):
        super().__init__()
        self.mamba = VSSBlock2D(dim=channels, d_state=d_state, expand=expand, dropout=dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(init_gamma))

        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, feat: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != feat.shape[-2:]:
            gate = F.interpolate(gate, size=feat.shape[-2:], mode="bilinear", align_corners=True)
        gate = gate.clamp(0.0, 1.0)

        # The gate enters Mamba internally, not only after Mamba output.
        mamba_feat = self.mamba(feat, gate=gate)
        delta = mamba_feat - feat

        strength = 1.0 + torch.sigmoid(self.alpha) * gate
        out = feat + torch.tanh(self.gamma) * delta * strength
        out = self.post(out)
        return out


# =========================================================
# 5. Multi-scale UGM-Mamba U-Net
# =========================================================

class EncoderMultiScaleUGMMamba(nn.Module):
    """
    Encoder with:
        x2 uncertainty head
        x2/x3/x4 uncertainty-calibrated Mamba modulation
        independent x2 CASM branch for class-wise anatomical state modeling
    """
    def __init__(
        self,
        params: Dict,
        use_mamba: bool = True,
        d_state_x2: int = 16,
        d_state_x3: int = 16,
        d_state_x4: int = 16,
        expand_x2: float = 2.0,
        expand_x3: float = 2.0,
        expand_x4: float = 2.0,
        detach_prob_for_gate: bool = True,
        use_casm: bool = True,
        casm_d_state: int = 8,
        casm_expand: float = 2.0,
        detach_prob_for_casm: bool = True,
    ):
        super().__init__()
        self.params = params
        self.in_chns = params["in_chns"]
        self.ft_chns = params["feature_chns"]
        self.n_class = params["class_num"]
        self.dropout = params["dropout"]
        assert len(self.ft_chns) == 5

        self.use_mamba = use_mamba
        self.use_casm = use_casm

        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

        self.x2_uncertainty_head = X2UncertaintyHead(
            in_channels=self.ft_chns[2],
            num_classes=self.n_class,
            detach_prob_for_gate=detach_prob_for_gate,
        )

        # Independent CASM branch at x2. It is separate from UC-Mamba and can be drawn as an independent module.
        if self.use_casm:
            self.casm_x2 = ClassWiseAnatomicalStateMamba(
                channels=self.ft_chns[2],
                num_classes=self.n_class,
                d_state=casm_d_state,
                expand=casm_expand,
                detach_prob=detach_prob_for_casm,
            )
            self.casm_beta = nn.Parameter(torch.tensor(0.05))
            self.casm_fuse = nn.Sequential(
                nn.Conv2d(self.ft_chns[2], self.ft_chns[2], kernel_size=1),
                nn.BatchNorm2d(self.ft_chns[2]),
                nn.GELU(),
            )
        else:
            self.casm_x2 = None
            self.casm_beta = None
            self.casm_fuse = None

        if self.use_mamba:
            self.ugm_x2 = ExternalGateUGMMamba(
                channels=self.ft_chns[2], d_state=d_state_x2, expand=expand_x2, dropout=0.0
            )
            self.ugm_x3 = ExternalGateUGMMamba(
                channels=self.ft_chns[3], d_state=d_state_x3, expand=expand_x3, dropout=0.0
            )
            self.ugm_x4 = ExternalGateUGMMamba(
                channels=self.ft_chns[4], d_state=d_state_x4, expand=expand_x4, dropout=0.0
            )
        else:
            self.ugm_x2 = None
            self.ugm_x3 = None
            self.ugm_x4 = None

    def forward(self, x: torch.Tensor) -> Tuple[Sequence[torch.Tensor], Dict[str, torch.Tensor]]:
        aux_outputs = {}

        x0 = self.in_conv(x)      # 256x256, 16
        x1 = self.down1(x0)       # 128x128, 32
        x2_raw = self.down2(x1)   # 64x64, 64

        coarse_logits_x2, gate2, aux_x2 = self.x2_uncertainty_head(x2_raw)
        aux_outputs.update(aux_x2)
        prob2 = aux_x2["prob_x2"]

        if self.ugm_x2 is not None:
            x2 = self.ugm_x2(x2_raw, gate2)
        else:
            x2 = x2_raw

        # Independent CASM branch: x2_raw + P2 -> class relation map A2 -> fuse back to x2.
        if self.casm_x2 is not None:
            casm_map, casm_aux = self.casm_x2(x2_raw, prob2)
            x2 = self.casm_fuse(x2 + torch.tanh(self.casm_beta) * casm_map)
            aux_outputs["casm_x2"] = casm_aux
            aux_outputs["casm_map_x2"] = casm_map

        x3_raw = self.down3(x2)   # 32x32, 128
        if self.ugm_x3 is not None:
            x3 = self.ugm_x3(x3_raw, gate2)
        else:
            x3 = x3_raw

        x4_raw = self.down4(x3)   # 16x16, 256
        if self.ugm_x4 is not None:
            x4 = self.ugm_x4(x4_raw, gate2)
        else:
            x4 = x4_raw

        # Save multi-scale gates for visualization.
        aux_outputs["gate_x3"] = F.interpolate(gate2, size=x3.shape[-2:], mode="bilinear", align_corners=True)
        aux_outputs["gate_x4"] = F.interpolate(gate2, size=x4.shape[-2:], mode="bilinear", align_corners=True)

        return [x0, x1, x2, x3, x4], aux_outputs


class Decoder(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.params = params
        self.ft_chns = params["feature_chns"]
        self.n_class = params["class_num"]
        assert len(self.ft_chns) == 5

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1, x2, x3, x4 = feature
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x_last = self.up4(x, x0)
        output = self.out_conv(x_last)
        return output, x_last


class UNet_MultiScale_UGM_Mamba(nn.Module):
    """
    U-Net with x2-generated multi-scale uncertainty-guided Mamba modulation.

    Recommended first setting for 25GB GPU:
        feature_chns=(16,32,64,128,256)
        d_state_x2=16, d_state_x3=16, d_state_x4=16
        expand_x2=2.0, expand_x3=2.0, expand_x4=2.0
    """
    def __init__(
        self,
        in_chns: int = 1,
        class_num: int = 4,
        feature_chns: Sequence[int] = (16, 32, 64, 128, 256),
        dropout: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
        use_mamba: bool = True,
        d_state_x2: int = 16,
        d_state_x3: int = 16,
        d_state_x4: int = 16,
        expand_x2: float = 2.0,
        expand_x3: float = 2.0,
        expand_x4: float = 2.0,
        detach_prob_for_gate: bool = True,
        use_casm: bool = True,
        casm_d_state: int = 8,
        casm_expand: float = 2.0,
        detach_prob_for_casm: bool = True,
    ):
        super().__init__()
        params = {
            "in_chns": in_chns,
            "feature_chns": list(feature_chns),
            "dropout": list(dropout),
            "class_num": class_num,
        }
        self.encoder = EncoderMultiScaleUGMMamba(
            params=params,
            use_mamba=use_mamba,
            d_state_x2=d_state_x2,
            d_state_x3=d_state_x3,
            d_state_x4=d_state_x4,
            expand_x2=expand_x2,
            expand_x3=expand_x3,
            expand_x4=expand_x4,
            detach_prob_for_gate=detach_prob_for_gate,
            use_casm=use_casm,
            casm_d_state=casm_d_state,
            casm_expand=casm_expand,
            detach_prob_for_casm=detach_prob_for_casm,
        )
        self.decoder = Decoder(params)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        feature, aux_outputs = self.encoder(x)
        output, x_last = self.decoder(feature)
        if return_aux:
            return output, aux_outputs
        return output
