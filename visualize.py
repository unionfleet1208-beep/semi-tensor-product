"""
visualize.py

论文图表生成脚本。运行方式：
  python visualize.py \
    --checkpoints results/stp_gamma4/best_model.pth results/bilinear_gamma4/best_model.pth \
    --gamma 4 \
    --test_list ./data/file_lists/llvip_test.txt \
    --output_dir ./paper_figures \
    --sample_idx 0 15 42   # 指定要可视化的图像序号

生成的图表：
  1. comparison_fig.png  ── 多方法并排对比图（含 ROI 放大框）
  2. intensity_profile.png ── ROI 扫描线强度曲线图
  3. gamma_ablation.png  ── 不同 γ 值下的指标折线图（消融实验）
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # 服务器上没有显示器时使用
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from datasets.fusion_dataset import FusionDataset
from models.network import build_network
from losses.fusion_loss import FusionLoss


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """加载模型并设置为 eval 模式。"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["cfg"]
    model = build_network(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def get_fusion_result(model, ir_lr: torch.Tensor, vis: torch.Tensor, device) -> np.ndarray:
    """运行一次前向传播，返回 [H, W] 的 numpy 灰度图（值域 0–1）。"""
    ir_lr = ir_lr.unsqueeze(0).to(device)  # [1, 1, H/γ, W/γ]
    vis = vis.unsqueeze(0).to(device)       # [1, 3, H, W]
    pred = model(ir_lr, vis)               # [1, 1, H, W]
    return pred.squeeze().cpu().numpy()    # [H, W]


def denormalize_vis(vis_tensor: torch.Tensor) -> np.ndarray:
    """
    将归一化后的可见光 tensor 还原为可视化的 numpy 图像。
    vis_tensor: [3, H, W]，ImageNet 归一化
    返回：[H, W, 3]，值域 0–1
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = vis_tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


# ─────────────────────────────────────────────
# 图 1：多方法并排对比图（含 ROI 放大）
# ─────────────────────────────────────────────

def plot_comparison(
    sample: dict,
    models_and_names: List[Tuple],  # [(model, name, device), ...]
    roi: Tuple[int, int, int, int], # (y1, x1, y2, x2)，ROI 的像素坐标
    save_path: str,
):
    """
    生成并排对比图。布局：
      上行：IR（低分辨率）| 可见光 | 各方法融合结果
      下行：对应的 ROI 放大图
    
    ROI 坐标建议选在包含行人/车辆热源边缘的区域，
    这里边缘锐利度的差异最为显著。
    """
    ir_lr   = sample["ir_lr"]    # [1, H/γ, W/γ]
    ir_hr   = sample["ir_hr"]    # [1, H, W]
    vis     = sample["vis"]      # [3, H, W]
    vis_raw = sample["vis_raw"]  # [3, H, W]

    # 准备各列的图像
    columns = []

    # 列 1：低分辨率红外（双线性插值放大仅用于显示，不进入网络）
    ir_lr_np = F.interpolate(ir_lr.unsqueeze(0), scale_factor=4, mode="nearest").squeeze().numpy()
    columns.append(("IR (low-res)", ir_lr_np, "gray"))

    # 列 2：可见光
    vis_np = denormalize_vis(vis)
    # 转为灰度（与融合结果同色域，便于公平比较）
    vis_gray = 0.299 * vis_raw[0] + 0.587 * vis_raw[1] + 0.114 * vis_raw[2]
    columns.append(("Visible", vis_gray.numpy(), "gray"))

    # 列 3 以后：各融合方法的结果
    for model, name, device in models_and_names:
        result = get_fusion_result(model, ir_lr, vis, device)
        columns.append((name, result, "gray"))

    n_cols = len(columns)
    y1, x1, y2, x2 = roi
    zoom = 2.5  # ROI 放大倍率（论文中通常放大 2–4×）

    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))
    plt.subplots_adjust(wspace=0.03, hspace=0.03)

    for col_idx, (title, img, cmap) in enumerate(columns):
        # 上行：完整图像
        ax_top = axes[0, col_idx]
        if cmap == "gray":
            ax_top.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            ax_top.imshow(img)

        # 在原图上画 ROI 红色矩形框
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor="red", facecolor="none"
        )
        ax_top.add_patch(rect)
        ax_top.set_title(title, fontsize=10, pad=3, fontweight="bold")
        ax_top.axis("off")

        # 下行：ROI 放大图
        ax_bot = axes[1, col_idx]
        roi_img = img[y1:y2, x1:x2] if img.ndim == 2 else img[y1:y2, x1:x2, :]
        if cmap == "gray":
            ax_bot.imshow(roi_img, cmap="gray", vmin=0, vmax=1,
                         interpolation="nearest")  # 关键：放大时用 nearest，避免引入平滑
        else:
            ax_bot.imshow(roi_img, interpolation="nearest")
        ax_bot.axis("off")
        # 在 ROI 放大图上也画红色边框（标明这是放大区域）
        for spine in ax_bot.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(2)
            spine.set_visible(True)

    # 在图像下方标注 ROI 放大说明
    fig.text(0.5, 0.01, f"Bottom row: ROI zoom ({zoom:.0f}×) of the red rectangle",
             ha="center", fontsize=8, color="gray")

    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"  对比图已保存：{save_path}")


# ─────────────────────────────────────────────
# 图 2：ROI 扫描线强度曲线
# ─────────────────────────────────────────────

def plot_intensity_profile(
    sample: dict,
    models_and_names: List[Tuple],
    roi: Tuple[int, int, int, int],  # ROI 区域
    scan_row_ratio: float = 0.5,     # 扫描线在 ROI 内的相对位置（0=顶部，1=底部）
    save_path: str = "intensity_profile.png",
):
    """
    绘制穿过 ROI 中某条水平扫描线的像素强度曲线。
    
    这是论文中最有说服力的定性可视化之一：
      - 插值方法在热源边缘处呈现"S 形"平缓过渡（热晕效应）
      - STP 方法在同一位置呈现接近垂直的跳变（锐利边缘）
    一目了然，不需要读者仔细比较图像细节。
    """
    y1, x1, y2, x2 = roi
    scan_row = y1 + int((y2 - y1) * scan_row_ratio)  # 扫描线的全图 y 坐标

    ir_lr = sample["ir_lr"]
    vis   = sample["vis"]

    # 颜色和线型设置（IEEE 双栏图通常要求黑白可读）
    styles = {
        "GT":       {"color": "#000000", "lw": 1.5, "ls": "-"},
        "STP (Ours)":   {"color": "#E63946", "lw": 2.0, "ls": "-"},
        "Bilinear": {"color": "#457B9D", "lw": 1.5, "ls": "--"},
        "Nearest":  {"color": "#2A9D8F", "lw": 1.5, "ls": "-."},
        "Deconv":   {"color": "#E9C46A", "lw": 1.5, "ls": ":"},
    }

    fig, ax = plt.subplots(figsize=(6, 3))

    # 绘制各方法在扫描线上的曲线
    for model, name, device in models_and_names:
        result = get_fusion_result(model, ir_lr, vis, device)  # [H, W]
        profile = result[scan_row, x1:x2]  # 扫描线上的像素值
        x_axis = np.arange(len(profile))
        style = styles.get(name, {"color": "gray", "lw": 1.0, "ls": "-"})
        ax.plot(x_axis, profile, label=name, **style)

    ax.set_xlabel("Pixel position in ROI", fontsize=10)
    ax.set_ylabel("Intensity", fontsize=10)
    ax.set_title(f"Intensity profile along scan line (y={scan_row})", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, x2 - x1)

    # 标注"热源边缘"位置（如果已知）
    # 可以在运行时手动添加：ax.axvline(x=edge_x, color='gray', ls='--', lw=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  强度曲线已保存：{save_path}")


# ─────────────────────────────────────────────
# 图 3：不同 γ 值下的指标折线图（消融实验）
# ─────────────────────────────────────────────

def plot_gamma_ablation(
    results: dict,  # {method_name: {gamma: {metric: value}}}
    metric: str,    # 要绘制的指标（"AG" / "EN" / "SSIM" 等）
    save_path: str,
):
    """
    绘制不同 γ 值下各方法的性能折线图。

    预期的视觉模式（由数学证明预测）：
      - 插值方法：随 γ 增大，性能下降明显（定理 6：误差随 γ 线性增长）
      - STP 方法：随 γ 增大，性能基本保持稳定（无插值误差，鲁棒性强）
    
    如果实验数据呈现这种模式，就完成了理论—实验的完整闭环。
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    gammas = sorted({g for method_results in results.values() for g in method_results.keys()})
    styles_by_method = {
        "STP (Ours)": {"color": "#E63946", "marker": "o", "lw": 2.0, "ls": "-"},
        "Bilinear":   {"color": "#457B9D", "marker": "s", "lw": 1.5, "ls": "--"},
        "Nearest":    {"color": "#2A9D8F", "marker": "^", "lw": 1.5, "ls": "-."},
        "Deconv":     {"color": "#E9C46A", "marker": "D", "lw": 1.5, "ls": ":"},
    }

    for method_name, gamma_results in results.items():
        values = [gamma_results.get(g, {}).get(metric, np.nan) for g in gammas]
        style = styles_by_method.get(method_name, {"color": "gray", "marker": "o", "lw": 1.0, "ls": "-"})
        ax.plot(gammas, values, label=method_name,
                markerfacecolor="white", markersize=6, **style)

    ax.set_xlabel("Resolution scale factor γ", fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f"{metric} vs. Resolution Scale Factor", fontsize=10)
    ax.set_xticks(gammas)
    ax.set_xticklabels([f"γ={g}" for g in gammas])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  消融实验图已保存：{save_path}")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成论文图表")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="模型权重路径列表（按显示顺序）")
    parser.add_argument("--names", nargs="+",
                        help="对应的方法名称（若不指定，从 checkpoint 的 cfg 中读取）")
    parser.add_argument("--test_list", default="./data/file_lists/llvip_test.txt")
    parser.add_argument("--output_dir", default="./paper_figures")
    parser.add_argument("--sample_idx", nargs="+", type=int, default=[0],
                        help="要可视化的测试集图像序号")
    parser.add_argument("--roi", nargs=4, type=int, default=[100, 200, 180, 340],
                        metavar=("Y1", "X1", "Y2", "X2"),
                        help="ROI 区域坐标（像素）。建议选择包含热源边缘的区域。")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载所有模型
    print("加载模型...")
    models_and_names = []
    for i, ckpt_path in enumerate(args.checkpoints):
        model, cfg = load_model(ckpt_path, device)
        name = args.names[i] if args.names and i < len(args.names) else cfg["model"]["fusion_method"]
        models_and_names.append((model, name, device))
        print(f"  已加载：{name}（{ckpt_path}）")

    # 加载测试集（只取需要的样本）
    # 为简化，直接从文件列表中取对应行
    with open(args.test_list) as f:
        all_pairs = [line.strip().split() for line in f if line.strip()]

    gamma = models_and_names[0][1]  # 取第一个模型的 gamma（消融实验中所有方法应相同）
    # 实际上 gamma 从 cfg 里拿
    gamma = torch.load(args.checkpoints[0], map_location="cpu")["cfg"]["model"]["gamma"]

    roi = tuple(args.roi)  # (y1, x1, y2, x2)

    # 对每个指定的样本生成图表
    criterion = FusionLoss()
    for idx in args.sample_idx:
        print(f"\n处理样本 #{idx}...")
        if idx >= len(all_pairs):
            print(f"  [跳过] 索引 {idx} 超出测试集大小 {len(all_pairs)}")
            continue

        ir_path, vis_path = all_pairs[idx]

        # 构造临时 dataset 加载单个样本
        tmp_dataset = FusionDataset(
            file_list_path=args.test_list,
            patch_size=None,   # 全图
            gamma=gamma,
            augment=False,
        )
        sample = tmp_dataset[idx]

        # 图 1：并排对比图
        plot_comparison(
            sample, models_and_names, roi,
            save_path=str(Path(args.output_dir) / f"comparison_sample{idx}.png")
        )

        # 图 2：强度曲线
        plot_intensity_profile(
            sample, models_and_names, roi,
            scan_row_ratio=0.5,
            save_path=str(Path(args.output_dir) / f"profile_sample{idx}.png")
        )

    print(f"\n所有图表已保存至：{args.output_dir}/")
    print("提示：--roi 参数需要根据实际图像内容手动调整，")
    print("      建议先用 matplotlib imshow 打开图像，找到热源边缘的像素坐标。")


if __name__ == "__main__":
    main()
