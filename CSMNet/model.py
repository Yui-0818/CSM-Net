"""CSM-Net：角膜荧光素染色形态网络（从零训练，纯卷积）。

结构要点：
- DRSB：并行 depthwise 3×3 / 5×5，分别处理点状与片状染色；
- Stain Gate（可选）：把荧光素图 S 注入每个 stage，而不是只做输入预处理；
- 多尺度池化：stage2/3/4 的 avg+max，stage4 再加 stain/boundary 加权；
- 主头 3 类；可选两级辅助头（0 vs 病、1 vs 2），推理只用主头。

最终提交配置是简化版 B：use_stain_gate=False、use_aux=False，
只保留“去红线 + S 输入 + DRSB + 多尺度池化 + 3 类主头”。
Stain Gate 与辅助头保留为消融开关，不作为最终结构。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth；训练时按样本随机丢弃整个残差分支。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) < keep
        return x * mask / keep


class StainGate(nn.Module):
    """用 S 的空间分布和特征通道均值生成空间门。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=1)
        # 初始 gate=0.5，乘 2 后为 1，残差块初始不改变特征尺度
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([s, pooled], dim=1)))
        return x * (2.0 * gate)


class SE(nn.Module):
    """通道重标定。"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class DRSB(nn.Module):
    """Dual-Receptive Stain Block。"""

    def __init__(self, channels: int, drop_path: float = 0.0,
                 use_stain_gate: bool = True):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1,
                             groups=channels, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2,
                             groups=channels, bias=False)
        self.bn5 = nn.BatchNorm2d(channels)
        self.fuse = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(channels)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn_proj = nn.BatchNorm2d(channels)
        self.se = SE(channels)
        self.gate = StainGate(channels) if use_stain_gate else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        # 残差分支末端零初始化，深层 block 初始为恒等映射
        nn.init.zeros_(self.bn_proj.weight)
        nn.init.zeros_(self.bn_proj.bias)

    def forward(self, x: torch.Tensor,
                s: torch.Tensor | None = None) -> torch.Tensor:
        a = F.silu(self.bn3(self.dw3(x)))
        b = F.silu(self.bn5(self.dw5(x)))
        y = F.silu(self.bn_fuse(self.fuse(torch.cat([a, b], dim=1))))
        y = self.bn_proj(self.proj(y))
        y = self.se(y)
        if self.gate is not None and s is not None:
            y = self.gate(y, s)
        return x + self.drop_path(y)


class Downsample(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=2, padding=1,
                            groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.bn1(self.dw(x)))
        return self.bn2(self.pw(x))


class Stage(nn.Module):
    def __init__(self, cin: int, cout: int, depth: int,
                 drop_path: float, use_stain_gate: bool):
        super().__init__()
        self.down = Downsample(cin, cout) if cin != cout else nn.Identity()
        self.blocks = nn.ModuleList([
            DRSB(cout, drop_path, use_stain_gate) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor,
                s: torch.Tensor | None = None) -> torch.Tensor:
        x = self.down(x)
        for blk in self.blocks:
            x = blk(x, s)
        return x


class CSMNet(nn.Module):
    def __init__(self, in_ch: int = 4, num_classes: int = 3,
                 widths: tuple[int, ...] = (64, 128, 192, 320),
                 depths: tuple[int, ...] = (2, 3, 3, 3),
                 drop_path: float = 0.05,
                 use_stain_gate: bool = True,
                 use_aux: bool = True,
                 head_dim: int = 256,
                 head_dropout: float = 0.3):
        super().__init__()
        assert len(widths) == 4 and len(depths) == 4
        self.in_ch = in_ch
        self.use_stain_gate = bool(use_stain_gate and in_ch == 4)
        self.use_aux = bool(use_aux)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, widths[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(),
        )

        total_blocks = sum(depths)
        dps = [drop_path * i / max(1, total_blocks - 1)
               for i in range(total_blocks)]
        k = 0
        stages = []
        cin = widths[0]
        for si, (cout, depth) in enumerate(zip(widths, depths)):
            stage_dps = dps[k:k + depth]
            k += depth
            # Stage 内每个 block 用自己的 drop path；这里取均值，简单可控
            stages.append(Stage(cin, cout, depth,
                                sum(stage_dps) / max(1, len(stage_dps)),
                                self.use_stain_gate))
            cin = cout
        self.stage1, self.stage2, self.stage3, self.stage4 = stages

        pool_dim = widths[1] * 2 + widths[2] * 2
        pool_dim += widths[3] * (4 if self.use_stain_gate else 2)
        self.head_dim = int(head_dim)
        self.fuse = nn.Sequential(
            nn.Linear(pool_dim, self.head_dim),
            nn.BatchNorm1d(self.head_dim),
            nn.SiLU(),
            nn.Dropout(float(head_dropout)),
        )
        self.main = nn.Linear(self.head_dim, num_classes)
        self.aux0 = nn.Linear(self.head_dim, 1) if self.use_aux else None
        self.aux12 = nn.Linear(self.head_dim, 1) if self.use_aux else None

        # Sobel kernel 不训练；用于边界加权池化
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]]) / 8.0
        ky = torch.tensor([[-1., -2., -1.],
                           [0., 0., 0.],
                           [1., 2., 1.]]) / 8.0
        self.register_buffer("sobel_x", kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", ky.view(1, 1, 3, 3), persistent=False)

    @staticmethod
    def _pool_avgmax(x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        return torch.cat([avg, mx], dim=1)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        s = x[:, 3:4] if self.in_ch == 4 else None
        x = self.stem(x)  # /4

        if s is not None and self.use_stain_gate:
            s1 = F.interpolate(s, size=x.shape[-2:], mode="bilinear",
                               align_corners=False)
        else:
            s1 = None
        f1 = self.stage1(x, s1)  # /4

        if s is not None and self.use_stain_gate:
            s2 = F.interpolate(s, size=(f1.shape[-2] // 2, f1.shape[-1] // 2),
                               mode="bilinear", align_corners=False)
        else:
            s2 = None
        f2 = self.stage2(f1, s2)  # /8

        if s is not None and self.use_stain_gate:
            s3 = F.interpolate(s, size=(f2.shape[-2] // 2, f2.shape[-1] // 2),
                               mode="bilinear", align_corners=False)
        else:
            s3 = None
        f3 = self.stage3(f2, s3)  # /16

        if s is not None and self.use_stain_gate:
            s4 = F.interpolate(s, size=(f3.shape[-2] // 2, f3.shape[-1] // 2),
                               mode="bilinear", align_corners=False)
        else:
            s4 = None
        f4 = self.stage4(f3, s4)  # /32

        pooled = [self._pool_avgmax(f2), self._pool_avgmax(f3)]
        p4 = self._pool_avgmax(f4)

        if s is not None and self.use_stain_gate:
            s4p = F.interpolate(s, size=f4.shape[-2:], mode="bilinear",
                                align_corners=False)
            w = s4p.clamp_min(0.0)
            p_stain = (f4 * w).sum(dim=(2, 3)) / w.sum(
                dim=(2, 3)).clamp_min(1e-6)

            gx = F.conv2d(s, self.sobel_x, padding=1)
            gy = F.conv2d(s, self.sobel_y, padding=1)
            b = torch.sqrt(gx * gx + gy * gy)
            b = b / b.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            b = b * s
            b4 = F.interpolate(b, size=f4.shape[-2:], mode="bilinear",
                               align_corners=False)
            wb = b4.clamp_min(0.0)
            p_boundary = (f4 * wb).sum(dim=(2, 3)) / wb.sum(
                dim=(2, 3)).clamp_min(1e-6)
            p4 = torch.cat([p4, p_stain, p_boundary], dim=1)

        feat = torch.cat(pooled + [p4], dim=1)
        z = self.fuse(feat)
        logits = self.main(z)
        if return_aux and self.use_aux:
            return logits, self.aux0(z), self.aux12(z)
        return logits
