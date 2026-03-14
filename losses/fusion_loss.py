"""
losses/fusion_loss.py

多项组合损失函数。

三个损失项各有不同的"频率敏感性"，彼此互补：
  L_pixel（像素损失）：L1 范数，惩罚整体亮度误差，对低频成分敏感。
                       选 L1 而不是 L2 的原因：L2 对亮点（强热源）过度惩罚，
                       容易使网络生成"保守"的平均值，丢失对比度。
                       
  L_percep（感知损失）：用预训练 VGG-19 的中间层特征计算误差，
                        对中高频纹理结构敏感。这一项"教"网络保留
                        人眼可感知的细节，而不是像素级精确匹配。
                        
  L_grad（梯度损失）：计算图像梯度（Sobel 算子）的 L1 误差，
                      专门惩罚边缘模糊。这是针对 STP 理论预测设计的：
                      定理 2 预测 STP 融合保留高频奇异值，
                      边缘清晰度应该更好——L_grad 直接把这个预测
                      变成可优化的训练信号。

权重建议：λ1=1.0, λ2=0.1, λ3=0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class PerceptualLoss(nn.Module):
    """
    基于 VGG-19 的感知损失。
    使用 relu2_2 层（第 2 个 maxpool 前的最后一个 relu）的特征图，
    该层对纹理细节敏感，且不会过于抽象（深层特征对融合任务过于语义化）。
    """

    def __init__(self):
        super().__init__()
        # 加载预训练 VGG-19，只保留前 9 层（到 relu2_2）
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:9])

        # 冻结所有参数：感知损失只提供梯度信号，不参与 VGG 权重更新
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # VGG 期望 RGB 三通道输入，融合结果是灰度图
        # 需要在前向传播中把灰度复制为三通道

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: 灰度图，shape [B, 1, H, W]，值域 [0, 1]
        Returns:
            标量损失值
        """
        # 灰度 → 伪 RGB（复制三份）
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)

        # 提取 VGG 特征（特征提取器已被 eval 模式冻结）
        feat_pred = self.feature_extractor(pred_rgb)
        feat_target = self.feature_extractor(target_rgb)

        return F.l1_loss(feat_pred, feat_target)


class GradientLoss(nn.Module):
    """
    图像梯度损失，使用 Sobel 算子估计水平和垂直梯度。

    这是验证 STP 理论优势最直接的损失：
    定理 2 预测 STP 保留了高频奇异值，
    而梯度损失惩罚任何边缘模糊，引导网络更好地利用这些被保留的高频信息。
    """

    def __init__(self):
        super().__init__()
        # Sobel 算子（不可学习，固定权重）
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        # 注册为 buffer（随模型移到 GPU，但不参与梯度计算）
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _get_gradient(self, img: torch.Tensor) -> torch.Tensor:
        """计算梯度幅值，shape [B, 1, H, W]。"""
        gx = F.conv2d(img, self.sobel_x, padding=1)
        gy = F.conv2d(img, self.sobel_y, padding=1)
        # 梯度幅值：sqrt(gx² + gy²)，加 1e-6 防止梯度爆炸
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad_pred = self._get_gradient(pred)
        grad_target = self._get_gradient(target)
        return F.l1_loss(grad_pred, grad_target)


class FusionLoss(nn.Module):
    """
    组合融合损失：L = λ1·L_pixel + λ2·L_percep + λ3·L_grad

    GT（监督信号）的选取策略：
      直接监督：target = 加权组合（0.5 * ir_hr + 0.5 * vis_gray）
      这是图像融合领域的标准 GT 构造方式：
        - ir_hr 提供热源轮廓信息
        - vis_gray（可见光灰度化）提供纹理细节
        - 0.5/0.5 的等权重是最中立的选择，避免先验地偏向某一模态
      注意：部分论文不用 GT，而是用"无监督"损失（互信息最大化），
            但对 IEEE 会议 5 页纸来说，有监督方法更容易控制实验变量。
    """

    def __init__(self, lambda1: float = 1.0, lambda2: float = 0.1, lambda3: float = 0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.perceptual = PerceptualLoss()
        self.gradient = GradientLoss()

    def build_target(self, ir_hr: torch.Tensor, vis_raw: torch.Tensor) -> torch.Tensor:
        """
        构建融合目标图像。
        Args:
            ir_hr:   高分辨率红外（GT），[B, 1, H, W]，值域 [0, 1]
            vis_raw: 未归一化可见光，[B, 3, H, W]，值域 [0, 1]
        Returns:
            target: [B, 1, H, W]，值域 [0, 1]
        """
        # 可见光转灰度（使用 ITU-R 601 亮度系数）
        vis_gray = (0.299 * vis_raw[:, 0:1] +
                    0.587 * vis_raw[:, 1:2] +
                    0.114 * vis_raw[:, 2:3])  # [B, 1, H, W]

        # 等权重融合作为 GT
        target = 0.5 * ir_hr + 0.5 * vis_gray
        return target.clamp(0, 1)

    def forward(
        self,
        pred: torch.Tensor,
        ir_hr: torch.Tensor,
        vis_raw: torch.Tensor,
    ) -> dict:
        """
        Args:
            pred:    网络输出的融合图像，[B, 1, H, W]，值域 [0, 1]
            ir_hr:   高分辨率红外 GT，[B, 1, H, W]
            vis_raw: 未归一化可见光，[B, 3, H, W]
        Returns:
            包含各损失项的字典（便于 TensorBoard 记录）
        """
        target = self.build_target(ir_hr, vis_raw)

        l_pixel = F.l1_loss(pred, target)
        l_percep = self.perceptual(pred, target)
        l_grad = self.gradient(pred, target)

        total = (self.lambda1 * l_pixel +
                 self.lambda2 * l_percep +
                 self.lambda3 * l_grad)

        return {
            "total":  total,
            "pixel":  l_pixel.detach(),
            "percep": l_percep.detach(),
            "grad":   l_grad.detach(),
        }
