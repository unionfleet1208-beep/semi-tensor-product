"""
models/fusion_modules.py

核心融合算子的实现，包括：
  - STPFusionModule：本文提出的半张量积融合模块（无插值）
  - BilinearFusion：双线性插值 + 拼接（基准方法 1）
  - NearestFusion：最近邻插值 + 拼接（基准方法 2）
  - DeconvFusion：转置卷积 + 拼接（基准方法 3，可学习上采样）

理解 STP 融合的关键思想：
  对于低分辨率红外特征 A ∈ R^{C×H×W} 和高分辨率可见光特征 B ∈ R^{C×γH×γW}，
  传统方法先把 A 插值到 [C×γH×γW]，再做拼接——这是有损的（见定理 1）。
  
  STP 方法的思路：把高分辨率空间分割成 H×W 个 γ×γ 的局部块，
  每个低分辨率红外位置 (i,j) 对应一个 γ×γ 的可见光局部块。
  用 Unfold 提取出这些块，再与对应的红外特征向量做代数乘法（非插值）。
  整个过程没有对红外特征做任何空间上的近似，满足定理 2 的保谱性质。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 本文方法：STP 融合模块
# ─────────────────────────────────────────────

class STPFusionModule(nn.Module):
    """
    基于半张量积的跨分辨率融合模块。

    数学对应：
      设低分辨率红外特征展平后为 A ∈ R^{C×n}（n = H*W），
      高分辨率可见光特征展平后为 B ∈ R^{C×p}（p = γ²n）。
      
      STP 扩展：A_hat = A ⊗ I_{γ²}，shape [C×γ² × n×γ²]。
      
      实现等价性：
        (A ⊗ I_{γ²}) 作用在 B^T 上，等价于：
        对 B 按 γ×γ 的步长做 Unfold，得到每个低分辨率位置对应的 γ×γ 可见光块；
        再与对应位置的红外特征向量做逐元素乘法（即 Kronecker 积与 delta 函数的作用）。
        这一操作保留了红外特征的全部奇异值结构（见定理 2），同时引入了可见光的高频纹理。

    Args:
        channels: 特征图通道数 C
        gamma:    分辨率倍率
    """

    def __init__(self, channels: int, gamma: int):
        super().__init__()
        self.gamma = gamma
        self.C = channels
        gamma2 = gamma * gamma  # γ²

        # 融合后的后处理卷积：[C×γ², H, W] → [C×γ², H, W]
        # 作用：让网络学习调整各子像素通道的权重，而不是硬编码 STP 的输出
        # groups=channels：按原始 C 通道独立处理，允许同一通道的 γ² 个子像素互相交换信息，
        # 比 groups=C*γ²（完全逐通道）保留更多高频空间结构
        self.post_conv = nn.Sequential(
            nn.Conv2d(channels * gamma2, channels * gamma2, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels * gamma2),
            nn.GELU(),
            # 逐点卷积（pointwise）：整合跨通道信息
            nn.Conv2d(channels * gamma2, channels * gamma2, 1),
            nn.BatchNorm2d(channels * gamma2),
            nn.GELU(),
        )

        # PixelShuffle：把 [B, C×γ², H, W] → [B, C, γH, γW]
        # 这是把 γ² 个子通道重新排列成空间上的高分辨率特征图
        # 注意：这里的 PixelShuffle 不是上采样，而是子像素重排列（spatial rearrangement），
        # 因为空间维度（H, W）已经是低分辨率，γ² 个子通道对应γ×γ个子像素
        self.pixel_shuffle = nn.PixelShuffle(gamma)

        # 可学习的融合门控：让网络自适应地决定乘法项（STP交互项）与加法项（可见光保留项）
        # 的相对权重，防止红外特征接近0时完全抹去可见光细节。
        # 初始化偏置为 0，使 Sigmoid 输出从 0.5 开始（对两种贡献等权重），训练稳定。
        _gate_conv = nn.Conv2d(channels * gamma2, channels * gamma2, 1)
        nn.init.zeros_(_gate_conv.bias)
        self.gate = nn.Sequential(_gate_conv, nn.Sigmoid())

        # 残差连接：把可见光特征直接加到 STP 输出上（保留高频纹理）
        # 需要 1×1 卷积对齐通道数（vis 特征是 [C, γH, γW]，输出也是 [C, γH, γW]）
        self.residual_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, feat_ir: torch.Tensor, feat_rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_ir:  低分辨率红外特征，shape [B, C, H, W]
            feat_rgb: 高分辨率可见光特征，shape [B, C, γH, γW]
        Returns:
            fused:    融合后的高分辨率特征，shape [B, C, γH, γW]
        """
        B, C, H, W = feat_ir.shape
        gamma = self.gamma
        pH, pW = gamma * H, gamma * W

        # ── 步骤 1：从高分辨率 RGB 特征中提取 γ×γ 局部块 ──
        # Unfold 操作：把 [B, C, γH, γW] 分割成 H*W 个 γ×γ 的块
        # 输出 shape：[B, C*γ*γ, H*W]
        # 其中 rgb_patches[:, c*γ²:(c+1)*γ², k] = 位置 k 对应的 c 通道 γ×γ 块（展平）
        rgb_patches = F.unfold(
            feat_rgb,
            kernel_size=gamma,
            stride=gamma,   # stride=gamma 保证每个块不重叠，精确对应一个低分辨率位置
            padding=0,
        )  # [B, C*γ², H*W]
        rgb_patches = rgb_patches.reshape(B, C, gamma * gamma, H * W)
        # rgb_patches: [B, C, γ², H*W]

        # ── 步骤 2：低分辨率红外特征展平 ──
        ir_flat = feat_ir.reshape(B, C, H * W)  # [B, C, H*W]

        # ── 步骤 3：STP 代数融合（带残差加法，防止 IR 特征为 0 时抹去可见光细节）──
        # 数学含义：对于每个位置 k 和通道 c：
        #   fused[b, c, s, k] = ir_flat[b, c, k] × rgb_patches[b, c, s, k]
        # 即：红外的第 c 通道特征与可见光 γ×γ 局部块的第 s 个子像素相乘。
        # 这正是 (a_k ⊗ I_{γ²}) 与 b 的相互作用：保留了 a_k 的全部信息，
        # 同时引入了可见光的精细纹理，没有任何插值近似。
        # 额外加入 rgb_patches 本身作为加性残差，保证当 IR 特征趋近 0 时
        # 可见光高频细节仍能传播到融合输出（有助于提升 AG / SF 指标）。
        ir_expanded = ir_flat.unsqueeze(2)          # [B, C, 1, H*W]
        interact = ir_expanded * rgb_patches         # [B, C, γ², H*W]（STP 交互项）
        fused = interact + rgb_patches               # [B, C, γ², H*W]（加性残差）

        # ── 步骤 4：重塑为空间特征图 ──
        # [B, C, γ², H*W] → [B, C*γ², H*W] → [B, C*γ², H, W]
        fused = fused.reshape(B, C * gamma * gamma, H * W)
        fused = fused.reshape(B, C * gamma * gamma, H, W)

        # ── 步骤 5：可学习的后处理 + 门控 ──
        # 门控机制：学习每个子像素通道的保留比例，进一步控制 STP 输出的强度
        fused_post = self.post_conv(fused)                      # [B, C*γ², H, W]
        gate = self.gate(fused_post)                            # [B, C*γ², H, W] ∈ (0,1)
        fused_gated = gate * fused_post + (1 - gate) * fused   # 自适应加权融合

        # ── 步骤 6：子像素重排为高分辨率特征图 ──
        fused_hr = self.pixel_shuffle(fused_gated)  # [B, C, γH, γW]

        # ── 步骤 7：残差连接（加入可见光高频细节）──
        fused_hr = fused_hr + self.residual_proj(feat_rgb)

        return fused_hr  # [B, C, γH, γW]


# ─────────────────────────────────────────────
# 基准方法 1：双线性插值 + 拼接
# ─────────────────────────────────────────────

class BilinearFusion(nn.Module):
    """
    基准方法：将低分辨率红外特征双线性上采样到高分辨率，然后与可见光特征拼接。
    这是最常见的跨分辨率融合做法，也是本文重点对比的"有损"方法。
    
    根据命题 1，上采样操作等价于右乘一个秩亏缺的插值算子，
    信息维度损失率 η ≥ 1 - 1/γ²。
    """

    def __init__(self, channels: int, gamma: int):
        super().__init__()
        self.gamma = gamma
        # 拼接后通道数变为 2C，用 1×1 卷积压缩回 C
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, feat_ir: torch.Tensor, feat_rgb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_ir.shape
        # 双线性插值上采样低分辨率红外特征
        ir_upsampled = F.interpolate(
            feat_ir,
            scale_factor=self.gamma,
            mode="bilinear",
            align_corners=False,
        )  # [B, C, γH, γW]

        # 通道维度拼接
        fused = torch.cat([ir_upsampled, feat_rgb], dim=1)  # [B, 2C, γH, γW]
        return self.fusion_conv(fused)  # [B, C, γH, γW]


# ─────────────────────────────────────────────
# 基准方法 2：最近邻插值 + 拼接
# ─────────────────────────────────────────────

class NearestFusion(nn.Module):
    """
    基准方法：最近邻插值上采样 + 拼接。
    最近邻比双线性更"粗糙"，但不引入双线性的平滑效应，
    作为另一个有损方法的下界对比。
    """

    def __init__(self, channels: int, gamma: int):
        super().__init__()
        self.gamma = gamma
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, feat_ir: torch.Tensor, feat_rgb: torch.Tensor) -> torch.Tensor:
        ir_upsampled = F.interpolate(
            feat_ir,
            scale_factor=self.gamma,
            mode="nearest",
        )  # [B, C, γH, γW]
        fused = torch.cat([ir_upsampled, feat_rgb], dim=1)
        return self.fusion_conv(fused)


# ─────────────────────────────────────────────
# 基准方法 3：转置卷积 + 拼接（可学习上采样）
# ─────────────────────────────────────────────

class DeconvFusion(nn.Module):
    """
    基准方法：转置卷积（反卷积）上采样 + 拼接。
    与双线性插值不同，转置卷积的上采样权重是可学习的，
    被视为"更强"的上采样基准，用于测试可学习插值是否能弥补有损性。
    
    根据命题 1，无论插值权重是否可学习，只要是线性上采样算子，
    rank(A @ U) ≤ n 的秩亏缺就必然存在，因为这是线性代数的性质，
    不依赖于 U 的具体数值。
    """

    def __init__(self, channels: int, gamma: int):
        super().__init__()
        self.gamma = gamma
        # 转置卷积：kernel_size=gamma, stride=gamma 恰好将空间尺寸扩大 gamma 倍
        self.deconv = nn.ConvTranspose2d(
            channels, channels,
            kernel_size=gamma,
            stride=gamma,
            padding=0,
        )
        self.bn_deconv = nn.BatchNorm2d(channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, feat_ir: torch.Tensor, feat_rgb: torch.Tensor) -> torch.Tensor:
        ir_upsampled = F.gelu(self.bn_deconv(self.deconv(feat_ir)))
        fused = torch.cat([ir_upsampled, feat_rgb], dim=1)
        return self.fusion_conv(fused)


# ─────────────────────────────────────────────
# 工厂函数：根据名称创建融合模块
# ─────────────────────────────────────────────

def build_fusion_module(name: str, channels: int, gamma: int) -> nn.Module:
    """
    根据名称创建对应的融合模块。
    统一的创建接口使得消融实验中切换方法只需修改配置文件的一个字段。
    """
    registry = {
        "stp":      STPFusionModule,
        "bilinear": BilinearFusion,
        "nearest":  NearestFusion,
        "deconv":   DeconvFusion,
    }
    if name not in registry:
        raise ValueError(f"未知融合方法：{name}，可选：{list(registry.keys())}")
    return registry[name](channels=channels, gamma=gamma)
