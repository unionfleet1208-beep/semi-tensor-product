"""
evaluate.py - 优化版：支持抽样评估与流式计算，防止内存溢出。
"""

import os
import sys
import random
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity as ski_ssim

# 把项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent))

from datasets.fusion_dataset import build_dataloader
from models.network import build_network

# ─────────────────────────────────────────────
# 各项指标的计算函数 (保持不变)
# ─────────────────────────────────────────────

def average_gradient(img: np.ndarray) -> float:
    gx = np.gradient(img, axis=1)
    gy = np.gradient(img, axis=0)
    mag = np.sqrt((gx ** 2 + gy ** 2) / 2.0)
    return float(np.mean(mag))

def spatial_frequency(img: np.ndarray) -> float:
    rf = np.sqrt(np.mean(np.diff(img, axis=1) ** 2))
    cf = np.sqrt(np.mean(np.diff(img, axis=0) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))

def entropy(img: np.ndarray) -> float:
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    hist, _ = np.histogram(img_uint8.flatten(), bins=256, range=(0, 255))
    hist = hist[hist > 0]
    prob = hist / hist.sum()
    return float(-np.sum(prob * np.log2(prob)))

def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return float(ski_ssim(pred, target, data_range=1.0))

def vif(pred_tensor: torch.Tensor, target_tensor: torch.Tensor) -> float:
    try:
        import piq
        # piq 期望 [B, C, H, W]
        val = piq.vif_p(pred_tensor, target_tensor, data_range=1.0)
        return float(val.item())
    except Exception:
        return float("nan")

# ─────────────────────────────────────────────
# 主评估流程
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(checkpoint_path: str, gamma: int, test_list: str, device: torch.device, sample_rate: int = 10):
    print(f"\n{'='*60}")
    print(f"开始评估：{checkpoint_path}")
    print(f"配置：γ={gamma}, 抽样率=1/{sample_rate}")
    print(f"{'='*60}")

    # ── 1. 加载模型 ──
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_network(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    method = cfg["model"]["fusion_method"]
    print(f"检测到训练方法：{method}")

    # ── 2. 构建 DataLoader ──
    # 注意：此处必须定义 test_loader
    test_loader = build_dataloader(
        file_list_path=test_list,
        batch_size=1,
        patch_size=None, # 全图评估
        gamma=gamma,
        augment=False,
        num_workers=4
    )
    
    total_samples = len(test_loader)
    print(f"数据集载入成功：共 {total_samples} 样本")

    # ── 3. 初始化累加器 ──
    from losses.fusion_loss import FusionLoss
    criterion = FusionLoss()
    
    metrics_sum = {"AG": 0.0, "EN": 0.0, "SF": 0.0, "SSIM": 0.0, "VIF": 0.0}
    evaluated_count = 0

    # ── 4. 推理与流式计算 ──
    # 固定随机种子抽样：每次运行从数据集中选出相同的 1/sample_rate 图片
    n_to_evaluate = max(1, total_samples // sample_rate)
    _rng = random.Random(42)
    selected_indices = set(_rng.sample(range(total_samples), n_to_evaluate))
    print(f"固定随机抽样（seed=42）：从 {total_samples} 张中选 {n_to_evaluate} 张")

    pbar = tqdm(total=n_to_evaluate, desc="评估进度")
    
    for i, batch in enumerate(test_loader):
        # 固定随机抽样逻辑
        if i not in selected_indices:
            continue
            
        ir_lr = batch["ir_lr"].to(device)
        ir_hr = batch["ir_hr"].to(device)
        vis   = batch["vis"].to(device)
        vis_raw = batch["vis_raw"].to(device)

        # 推理
        pred = model(ir_lr, vis) 
        target = criterion.build_target(ir_hr, vis_raw)

        # 立即转为 Numpy 释放显存/内存占用
        pred_cpu = pred.cpu()
        target_cpu = target.cpu()
        pred_np = pred_cpu.squeeze().numpy()
        target_np = target_cpu.squeeze().numpy()

        # 计算指标并累加
        metrics_sum["AG"]   += average_gradient(pred_np)
        metrics_sum["EN"]   += entropy(pred_np)
        metrics_sum["SF"]   += spatial_frequency(pred_np)
        metrics_sum["SSIM"] += ssim(pred_np, target_np)
        
        v_val = vif(pred_cpu, target_cpu)
        if not np.isnan(v_val):
            metrics_sum["VIF"] += v_val

        evaluated_count += 1
        pbar.update(1)

        # 每 50 张清理一次显存碎片
        if evaluated_count % 50 == 0:
            torch.cuda.empty_cache()

    pbar.close()

    # ── 5. 计算均值并打印 ──
    if evaluated_count == 0:
        print("未评估任何图像，请检查 sample_rate 或数据集。")
        return

    final_results = {k: v / evaluated_count for k, v in metrics_sum.items()}

    print(f"\n{'─'*40}")
    print(f"{'指标 (抽样平均)':<15} {'结果':>15}")
    print(f"{'─'*40}")
    for k, v in final_results.items():
        if k == "VIF" and v == 0: continue
        print(f"  {k:<13}  {v:>15.4f}")
    print(f"{'─'*40}")
    print(f"实际评估样本数: {evaluated_count}")

    # ── 6. 保存结果 ──
    save_dir = Path(checkpoint_path).parent
    result_path = save_dir / f"test_results_sampled_x{sample_rate}.json"
    with open(result_path, "w") as f:
        save_data = {
            "method": method, 
            "gamma": gamma, 
            "sample_rate": sample_rate,
            "evaluated_count": evaluated_count,
            **final_results
        }
        json.dump(save_data, f, indent=2)
    print(f"结果已保存至：{result_path}")

def main():
    parser = argparse.ArgumentParser(description="STP 融合网络抽样评估脚本")
    parser.add_argument("--checkpoint", required=True, help="模型路径")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--sample_rate", type=int, default=10, help="每隔多少张测一张")
    parser.add_argument("--test_list", default="./data/file_lists/llvip_test.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluate(
        checkpoint_path=args.checkpoint,
        gamma=args.gamma,
        test_list=args.test_list,
        device=device,
        sample_rate=args.sample_rate
    )

if __name__ == "__main__":
    main()