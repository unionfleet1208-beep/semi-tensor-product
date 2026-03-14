"""
models/network.py

完整的双流融合网络。

网络由三个部分组成：
  1. 双流特征提取器（Dual-stream Encoder）
     - IR 分支：处理低分辨率红外输入，输出低分辨率深层特征
     - RGB 分支：处理高分辨率可见光输入，输出高分辨率浅层特征
     - 两个分支使用相同的网络结构，但权重不共享（因为两种模态的特性不同）
     
  2. 融合模块（Fusion Module）
     - 支持 STP / Bilinear / Nearest / Deconv 四种方式
     - 输出高分辨率融合特征 [B, C, γH, γW]
     
  3. 解码器（Decoder）
     - 将融合特征解码为单通道融合图像（灰度）
     - 使用跳跃连接引入可见光的浅层纹理

设计原则：
  编码器尽量"公平"——四种融合方法的编码器权重完全相同，
  这样消融实验中只有融合算子本身的差异是变量，其余均控制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fusion_modules import build_fusion_module


# ─────────────────────────────────────────────
# 基础模块：残差块
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    标准残差块。由两个 3×3 卷积 + BN + GELU 组成，
    附带一个跳跃连接（skip connection）。
    选择 GELU 而不是 ReLU：在特征融合任务中 GELU 通常给出更平滑的梯度。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ─────────────────────────────────────────────
# 特征提取器（编码器分支）
# ─────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    单模态特征提取器，被 IR 分支和 RGB 分支复用（但权重不共享）。

    网络结构：
      输入 → Conv_in（调整通道数）→ ResBlock × num_blocks → 输出特征
    
    注意：此模块不做任何下采样（stride=1）。
    这意味着 IR 分支输入 [B, 1, H/γ, W/γ]，输出也是 [B, C, H/γ, W/γ]；
    RGB 分支输入 [B, 3, H, W]，输出也是 [B, C, H, W]。
    分辨率差异由融合模块处理，而不是在编码器中消除。

    Args:
        in_channels:  输入通道数（IR 为 1，RGB 为 3）
        channels:     中间特征通道数 C
        num_blocks:   残差块数量
    """

    def __init__(self, in_channels: int, channels: int = 32, num_blocks: int = 4):
        super().__init__()
        # 输入卷积：将输入通道映射为特征通道，并做一次非线性变换
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # 主干：堆叠残差块
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.conv_in(x))


# ─────────────────────────────────────────────
# 解码器
# ─────────────────────────────────────────────

class FusionDecoder(nn.Module):
    """
    将高分辨率融合特征解码为单通道融合图像。

    设计思路：
      融合模块已经输出了 [B, C, γH, γW] 的高分辨率特征，
      解码器只需要做通道压缩 + 精细调整，不再涉及空间分辨率变化。
      最后一层用 Sigmoid 把输出压到 [0, 1]，对应归一化后的灰度图。
    """

    def __init__(self, channels: int, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)]
        )
        # 输出层：C 通道 → 1 通道（融合后的灰度图）
        self.conv_out = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid(),  # 输出压到 [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.blocks(x))


# ─────────────────────────────────────────────
# 完整融合网络
# ─────────────────────────────────────────────

class FusionNetwork(nn.Module):
    """
    完整的双流融合网络，整合编码器、融合模块和解码器。

    Args:
        fusion_method:   融合方法名称（'stp'|'bilinear'|'nearest'|'deconv'）
        channels:        特征通道数 C（默认 32，论文中可调到 64）
        gamma:           分辨率倍率
        enc_blocks:      编码器残差块数量
        dec_blocks:      解码器残差块数量
    """

    def __init__(
        self,
        fusion_method: str = "stp",
        channels: int = 32,
        gamma: int = 4,
        enc_blocks: int = 4,
        dec_blocks: int = 2,
    ):
        super().__init__()
        self.gamma = gamma
        self.fusion_method = fusion_method

        # IR 分支编码器：输入单通道（灰度红外），在低分辨率空间提取特征
        self.ir_encoder = FeatureExtractor(
            in_channels=1, channels=channels, num_blocks=enc_blocks
        )

        # RGB 分支编码器：输入三通道可见光，在高分辨率空间提取特征
        self.rgb_encoder = FeatureExtractor(
            in_channels=3, channels=channels, num_blocks=enc_blocks
        )

        # 融合模块（可替换，是消融实验的核心变量）
        self.fusion = build_fusion_module(fusion_method, channels, gamma)

        # 解码器
        self.decoder = FusionDecoder(channels=channels, num_blocks=dec_blocks)

    def forward(self, ir_lr: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ir_lr: 低分辨率红外图像，shape [B, 1, H, W]
            vis:   高分辨率可见光图像，shape [B, 3, γH, γW]（已归一化）
        Returns:
            fused: 融合后的灰度图，shape [B, 1, γH, γW]，值域 [0, 1]
        """
        # 双流特征提取（各自在本模态的分辨率空间提取特征）
        feat_ir = self.ir_encoder(ir_lr)   # [B, C, H, W]
        feat_rgb = self.rgb_encoder(vis)   # [B, C, γH, γW]

        # 融合（核心步骤，根据方法不同行为不同）
        feat_fused = self.fusion(feat_ir, feat_rgb)  # [B, C, γH, γW]

        # 解码为图像
        fused_img = self.decoder(feat_fused)  # [B, 1, γH, γW]

        return fused_img

    def count_parameters(self) -> int:
        """计算可训练参数总量，写入论文 Implementation Details。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_network(cfg: dict) -> FusionNetwork:
    """根据配置文件创建网络实例。"""
    return FusionNetwork(
        fusion_method=cfg.get("fusion_method", "stp"),
        channels=cfg.get("channels", 32),
        gamma=cfg.get("gamma", 4),
        enc_blocks=cfg.get("enc_blocks", 4),
        dec_blocks=cfg.get("dec_blocks", 2),
    )
