"""
datasets/fusion_dataset.py

PyTorch Dataset 类，负责加载图像对并构造跨分辨率场景。

设计原则：
  - 红外图像经过 Bicubic 降采样至原始尺寸的 1/gamma（模拟廉价红外传感器）
  - 可见光图像保持原始高分辨率（模拟高端 RGB 相机）
  - 两路图像的几何增广（裁剪、翻转）必须同步，保证像素级配准
  - 同时返回高分辨率红外图（用于部分指标的 GT 计算）
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF


class FusionDataset(torch.utils.data.Dataset):
    """
    跨分辨率红外/可见光图像融合数据集。

    Args:
        file_list_path: 每行格式为 "ir_path vis_path" 的文本文件路径
        patch_size:     训练时随机裁剪的 patch 尺寸（测试时设为 None 使用全图）
        gamma:          分辨率倍率，红外被降采样至 1/gamma
        augment:        是否进行数据增广（训练时开启，测试时关闭）
        gamma_list:     若不为 None，每次取样时随机从列表中选择 gamma 值
                        用于训练时同时覆盖多个倍率（增强鲁棒性）
    """

    def __init__(
        self,
        file_list_path: str,
        patch_size: Optional[int] = 256,
        gamma: int = 4,
        augment: bool = True,
        gamma_list: Optional[List[int]] = None,
    ):
        self.patch_size = patch_size
        self.gamma = gamma
        self.augment = augment
        self.gamma_list = gamma_list  # 若设置，则随机选 gamma（训练时增强鲁棒性）

        # 读取文件列表
        self.pairs: List[Tuple[str, str]] = []
        with open(file_list_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        self.pairs.append((parts[0], parts[1]))

        if len(self.pairs) == 0:
            raise ValueError(f"文件列表为空：{file_list_path}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        ir_path, vis_path = self.pairs[idx]

        # 加载图像
        # 红外图像通常是灰度或伪彩色，统一转为灰度
        ir_img = Image.open(ir_path).convert("L")   # [H, W]，单通道
        vis_img = Image.open(vis_path).convert("RGB")  # [H, W, 3]

        # 如果启用随机 gamma 列表，每次随机选一个倍率
        gamma = random.choice(self.gamma_list) if self.gamma_list else self.gamma

        # --- 几何增广（必须同步）---
        if self.patch_size is not None:
            # 随机裁剪：先确定裁剪参数，再同时应用到两张图
            # 注意：裁剪参数基于可见光图像的尺寸确定
            i, j, h, w = self._get_crop_params(vis_img, self.patch_size)
            ir_img = TF.crop(ir_img, i, j, h, w)
            vis_img = TF.crop(vis_img, i, j, h, w)

        if self.augment:
            # 随机水平翻转（同步）
            if random.random() > 0.5:
                ir_img = TF.hflip(ir_img)
                vis_img = TF.hflip(vis_img)
            # 随机垂直翻转（同步）
            if random.random() > 0.5:
                ir_img = TF.vflip(ir_img)
                vis_img = TF.vflip(vis_img)

        # --- 转为 Tensor ---
        # 值域归一化到 [0, 1]
        ir_hr = TF.to_tensor(ir_img)   # [1, H, W]，高分辨率红外（作为 GT）
        vis_tensor = TF.to_tensor(vis_img)  # [3, H, W]

        # --- 构造低分辨率红外（核心步骤）---
        # 使用 Bicubic 降采样，而非双线性，原因：
        #   Bicubic 在频域上更接近真实传感器的低通特性（截止更陡峭），
        #   避免引入人为的高频混叠干扰实验结论
        H, W = ir_hr.shape[1], ir_hr.shape[2]
        target_h = H // gamma
        target_w = W // gamma

        # F.interpolate 要求 4D 输入：[B, C, H, W]
        ir_lr = F.interpolate(
            ir_hr.unsqueeze(0),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,  # False 是 PyTorch 推荐设置，避免边缘失真
        ).squeeze(0).clamp(0, 1)  # clamp 防止 Bicubic 产生轻微越界值

        # --- 可见光归一化 ---
        # 使用 ImageNet 统计量，与预训练特征提取器保持一致
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        vis_norm = (vis_tensor - mean) / std

        return {
            "ir_lr": ir_lr,        # [1, H//gamma, W//gamma]  低分辨率红外（网络输入）
            "ir_hr": ir_hr,        # [1, H, W]               高分辨率红外（GT，计算指标用）
            "vis": vis_norm,       # [3, H, W]               高分辨率可见光（网络输入）
            "vis_raw": vis_tensor, # [3, H, W]               未归一化可见光（可视化用）
            "gamma": gamma,        # int                      本样本使用的倍率
        }

    @staticmethod
    def _get_crop_params(img: Image.Image, patch_size: int) -> Tuple[int, int, int, int]:
        """获取随机裁剪参数（i, j, h, w）。"""
        W, H = img.size  # PIL 返回 (width, height)
        if H < patch_size or W < patch_size:
            # 图像小于 patch_size 时，先 resize 再裁剪
            scale = max(patch_size / H, patch_size / W) + 0.1
            H, W = int(H * scale), int(W * scale)

        i = random.randint(0, H - patch_size)
        j = random.randint(0, W - patch_size)
        return i, j, patch_size, patch_size


def build_dataloader(
    file_list_path: str,
    batch_size: int,
    patch_size: Optional[int],
    gamma: int,
    augment: bool,
    num_workers: int = 4,
    gamma_list: Optional[List[int]] = None,
) -> torch.utils.data.DataLoader:
    """
    构建 DataLoader 的工厂函数，统一管理所有参数。
    训练时：patch_size=256, augment=True, shuffle=True
    测试时：patch_size=None, augment=False, shuffle=False
    """
    dataset = FusionDataset(
        file_list_path=file_list_path,
        patch_size=patch_size,
        gamma=gamma,
        augment=augment,
        gamma_list=gamma_list,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,       # 训练时打乱，测试时不打乱
        num_workers=num_workers,
        pin_memory=True,       # 加速 CPU→GPU 数据传输
        drop_last=augment,     # 训练时丢弃不完整的最后一个 batch
        persistent_workers=(num_workers > 0),
    )
    print(f"  DataLoader 构建完成：{len(dataset)} 样本，"
          f"batch_size={batch_size}，共 {len(loader)} 个 batch")
    return loader
