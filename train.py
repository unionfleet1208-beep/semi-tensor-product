"""
train.py

运行方式：
  python train.py --method stp        # 单独训练一组
  python train.py --run_all           # 一键跑完全部四组消融
  python train.py --method stp --override model.gamma=2

超参数文献来源（均从原始论文/代码直接读取，非推测）：
  lr=0.001        ← DSA-Net 原文（双流单阶段 CNN，与本文结构最相近）
  Adam            ← DSA-Net 原文
  epochs=25       ← DSA-Net 原文
  batch=8         ← CDDFuse 原始代码（DSA-Net 的 batch=29 受显存限制，改 2 的幂次）
  patch=128       ← CDDFuse 官方代码（DSA-Net 用 120，差异可忽略）
  StepLR(10,0.5)  ← 补充衰减，防止 lr=0.001 在中后期震荡（DSA-Net 无此项）
  clip_grad=0.01  ← CDDFuse 原始代码直接读取的值
  训练 200 对     ← CDDFuse/MGFusion 领域标准协议
  验证 200 对     ← 从剩余数据取，seed 与训练不同，保证无重叠

论文 Implementation Details 可写：
  "We follow the training protocol of DSA-Net: Adam, lr=0.001, batch=8, 25 epochs.
   Patch size is 128×128 following CDDFuse. Gradient clipping max_norm=0.01
   following CDDFuse. StepLR (step=10, γ=0.5) is added for stability.
   200 training pairs are randomly sampled from LLVIP following CDDFuse/MGFusion."
"""

import os, sys, time, argparse, logging, yaml
from pathlib import Path
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))
from datasets.fusion_dataset import build_dataloader
from models.network import build_network
from losses.fusion_loss import FusionLoss
from evaluate import compute_metrics


# ──────────────────────────────────────────────────────────────
# 超参数配置（四组消融共用，保证公平对比）
# ──────────────────────────────────────────────────────────────

BASE_CFG = {
    "data": {
        "train_list":  "./data/file_lists/llvip_train_200.txt",   # 200 对训练
        "val_list":    "./data/file_lists/llvip_val_200.txt",     # 200 对验证（新增）
        "test_list":   "./data/file_lists/llvip_test.txt",        # 官方 3463 对测试
        "patch_size":  128,       # CDDFuse 官方代码标准
        "num_workers": 4,
    },
    "model": {
        "fusion_method": "stp",
        "channels":   32,
        "gamma":      4,          # LLVIP: IR(320×256) vs Vis(1280×1024) ≈ 4×
        "enc_blocks": 4,
        "dec_blocks": 2,
    },
    "loss": {
        "lambda1": 1.0,   # 像素 L1
        "lambda2": 0.1,   # 感知损失
        "lambda3": 0.5,   # 梯度损失（对应 STP 保谱定理的直接监督）
    },
    "train": {
        "batch_size": 8,          # CDDFuse 原始代码
        "epochs":     25,         # DSA-Net 原文
        "lr":         1e-3,       # DSA-Net 原文 (0.001)
        "lr_step":    10,         # StepLR step：补充衰减
        "lr_gamma":   0.5,        # StepLR gamma：每次减半
        "clip_grad":  0.01,       # CDDFuse 原始代码直接读取值
        "save_every": 5,          # 每 5 epoch 保存（共 25 epoch，存 5 次）
        "log_every":  20,
        "val_every":  1,
    },
    "output": {
        "save_dir": "./results",
        "exp_name": "stp",
    },
}

# 四组消融：唯一变量是 fusion_method，其余全部相同
ABLATION_GROUPS = [
    {
        "fusion_method": "bilinear",
        "exp_name":      "ablation_bilinear",
        "desc":          "对照组A - Bilinear插值+拼接（现有方法隐含预处理的显式化）",
    },
    {
        "fusion_method": "nearest",
        "exp_name":      "ablation_nearest",
        "desc":          "对照组B - 最近邻插值+拼接",
    },
    {
        "fusion_method": "deconv",
        "exp_name":      "ablation_deconv",
        "desc":          "对照组C - 转置卷积+拼接（可学习插值）",
    },
    {
        "fusion_method": "stp",
        "exp_name":      "ablation_stp",
        "desc":          "实验组  - STP融合（本文方法，无插值）",
    },
]


# ──────────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────────

def setup_logging(save_dir, exp_name):
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter(f"[{exp_name}][%(asctime)s] %(message)s", "%H:%M:%S")
    for h in [logging.StreamHandler(),
              logging.FileHandler(os.path.join(save_dir, "train_log.txt"))]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def make_pair_list(full_path, out_path, n=200, seed=42):
    """
    从完整列表中固定随机采样 n 对，写入 out_path。
    文件已存在则跳过，保证四组消融用完全相同的样本。

    训练集：seed=42
    验证集：seed=99（与训练集不同，保证无重叠）
    两个 seed 不同但从同一个 full_path 采样，因此理论上可能有少量重叠。
    为彻底避免重叠，先取训练集，再从剩余部分取验证集——见 prepare_data_lists()。
    """
    if Path(out_path).exists():
        return
    import random
    random.seed(seed)
    with open(full_path) as f:
        lines = [l for l in f if l.strip()]
    sampled = random.sample(lines, min(n, len(lines)))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(sampled)
    print(f"  [数据] 已生成 {n} 对列表 → {out_path}  (seed={seed})")


def prepare_data_lists(cfg):
    """
    确保训练集和验证集列表文件都已生成，且两者无重叠。

    策略：
      1. 先从完整训练列表随机取 200 对作为训练集（seed=42）
      2. 再从剩余的图像对中随机取 200 对作为验证集（seed=99）
      这样严格保证两者不重叠。
    """
    import random

    train_out = cfg["data"]["train_list"]
    val_out   = cfg["data"]["val_list"]

    # 推断完整训练列表路径（去掉 _200 后缀）
    full_train = train_out.replace("_200", "")
    if not Path(full_train).exists():
        print(f"  [警告] 完整训练列表不存在：{full_train}")
        print(f"         请先运行 python scripts/01_prepare_data.py")
        return

    if Path(train_out).exists() and Path(val_out).exists():
        return  # 两个文件都已存在，跳过

    with open(full_train) as f:
        all_lines = [l for l in f if l.strip()]

    # 训练集：取 200 对
    random.seed(42)
    train_lines = random.sample(all_lines, min(200, len(all_lines)))

    # 验证集：从剩余中取 200 对（严格无重叠）
    remaining = [l for l in all_lines if l not in set(train_lines)]
    random.seed(99)
    val_lines = random.sample(remaining, min(200, len(remaining)))

    Path(train_out).parent.mkdir(parents=True, exist_ok=True)

    if not Path(train_out).exists():
        with open(train_out, "w") as f:
            f.writelines(train_lines)
        print(f"  [数据] 训练集：{len(train_lines)} 对 → {train_out}")

    if not Path(val_out).exists():
        with open(val_out, "w") as f:
            f.writelines(val_lines)
        print(f"  [数据] 验证集：{len(val_lines)} 对 → {val_out}")


# ──────────────────────────────────────────────────────────────
# 训练 / 验证
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    device, epoch, cfg, logger, writer):
    model.train()
    total, n = 0.0, len(loader)

    for i, batch in enumerate(loader):
        ir_lr   = batch["ir_lr"].to(device, non_blocking=True)
        ir_hr   = batch["ir_hr"].to(device, non_blocking=True)
        vis     = batch["vis"].to(device, non_blocking=True)
        vis_raw = batch["vis_raw"].to(device, non_blocking=True)

        pred   = model(ir_lr, vis)
        losses = criterion(pred, ir_hr, vis_raw)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        # clip_grad=0.01，来自 CDDFuse 原始代码
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["train"]["clip_grad"]
        )
        optimizer.step()

        total += losses["total"].item()
        if (i + 1) % cfg["train"]["log_every"] == 0 or i == n - 1:
            step = epoch * n + i
            logger.info(
                f"Ep{epoch:02d} {i+1:03d}/{n} | "
                f"loss={losses['total'].item():.4f} "
                f"px={losses['pixel'].item():.3f} "
                f"pe={losses['percep'].item():.3f} "
                f"gr={losses['grad'].item():.3f}"
            )
            for k in ["total", "pixel", "percep", "grad"]:
                writer.add_scalar(f"loss/{k}", losses[k].item(), step)

    return total / n


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, logger, writer):
    model.eval()
    val_loss, preds, targets = 0.0, [], []

    for batch in loader:
        ir_lr   = batch["ir_lr"].to(device)
        ir_hr   = batch["ir_hr"].to(device)
        vis     = batch["vis"].to(device)
        vis_raw = batch["vis_raw"].to(device)
        pred    = model(ir_lr, vis)
        losses  = criterion(pred, ir_hr, vis_raw)
        val_loss += losses["total"].item()
        preds.append(pred.cpu())
        targets.append(criterion.build_target(ir_hr, vis_raw).cpu())

    val_loss /= len(loader)
    # 验证时用全图（不裁剪），取前 100 张计算指标，速度与精度折中
    m = compute_metrics(torch.cat(preds)[:100], torch.cat(targets)[:100])
    logger.info(
        f"[Val] Ep{epoch:02d} loss={val_loss:.4f} | "
        f"AG={m['AG']:.4f} EN={m['EN']:.4f} SSIM={m['SSIM']:.4f}"
    )
    writer.add_scalar("val/loss", val_loss, epoch)
    for k, v in m.items():
        writer.add_scalar(f"val/{k}", v, epoch)
    return m["AG"]


# ──────────────────────────────────────────────────────────────
# 单组训练主流程
# ──────────────────────────────────────────────────────────────

def run_single(cfg):
    exp  = cfg["output"]["exp_name"]
    sdir = Path(cfg["output"]["save_dir"]) / exp
    for d in [sdir, sdir/"checkpoints", sdir/"tensorboard"]:
        d.mkdir(parents=True, exist_ok=True)

    with open(sdir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger = setup_logging(str(sdir), exp)
    writer = SummaryWriter(log_dir=str(sdir / "tensorboard"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"方法={cfg['model']['fusion_method']}  γ={cfg['model']['gamma']}  device={device}")
    logger.info(
        f"lr={cfg['train']['lr']}(DSA-Net)  "
        f"bs={cfg['train']['batch_size']}(CDDFuse)  "
        f"ep={cfg['train']['epochs']}(DSA-Net)  "
        f"patch={cfg['data']['patch_size']}(CDDFuse)  "
        f"clip={cfg['train']['clip_grad']}(CDDFuse)"
    )
    logger.info("=" * 60)

    # 确保训练/验证列表已生成且无重叠
    prepare_data_lists(cfg)

    gamma = cfg["model"]["gamma"]
    train_loader = build_dataloader(
        cfg["data"]["train_list"],
        cfg["train"]["batch_size"],
        cfg["data"]["patch_size"],
        gamma, augment=True,
        num_workers=cfg["data"]["num_workers"],
    )
    # 验证时不裁剪（patch_size=None），用全图评估
    val_loader = build_dataloader(
        cfg["data"]["val_list"],
        batch_size=1,
        patch_size=None,
        gamma=gamma, augment=False,
        num_workers=cfg["data"]["num_workers"],
    )
    logger.info(
        f"train={len(train_loader.dataset)}对 "
        f"({len(train_loader)}batch/ep)  "
        f"val={len(val_loader.dataset)}对(全图)"
    )

    model     = build_network(cfg["model"]).to(device)
    criterion = FusionLoss(**cfg["loss"]).to(device)
    logger.info(f"参数量：{model.count_parameters():,}")

    # Adam + lr=0.001（来自 DSA-Net 原文）
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=(0.9, 0.999),
        weight_decay=0,        # DSA-Net/CDDFuse 均不加 weight_decay
    )

    # StepLR(step=10, gamma=0.5)：在 DSA-Net 基础上补充的衰减
    # lr 轨迹：1e-3(ep1) → 5e-4(ep10) → 2.5e-4(ep20) → 1.25e-4(ep25结束)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["train"]["lr_step"],
        gamma=cfg["train"]["lr_gamma"],
    )

    best_ag = 0.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, cfg, logger, writer,
        )
        scheduler.step()
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        if epoch % cfg["train"]["val_every"] == 0:
            ag = validate(model, val_loader, criterion, device, epoch, logger, writer)
            if ag > best_ag:
                best_ag = ag
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_ag":              best_ag,
                    "cfg":                  cfg,
                }, sdir / "best_model.pth")
                logger.info(f"  ★ 最佳模型更新  AG={best_ag:.4f}")

        if epoch % cfg["train"]["save_every"] == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "cfg":                  cfg,
            }, sdir / "checkpoints" / f"epoch_{epoch:04d}.pth")

        elapsed = time.time() - t0
        logger.info(
            f"Ep{epoch:02d} done | {elapsed:.1f}s | "
            f"train_loss={train_loss:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}\n"
        )

    writer.close()
    logger.info(f"完成！最佳验证 AG={best_ag:.4f}  →  {sdir/'best_model.pth'}\n")
    return best_ag


# ──────────────────────────────────────────────────────────────
# 四组消融批量运行
# ──────────────────────────────────────────────────────────────

def run_all():
    """
    依次训练四组消融实验，结束后打印汇总表。

    控制变量（写进论文 Implementation Details）：
      训练数据：相同的 200 对（从完整列表 seed=42 采样，保证四组完全一致）
      验证数据：相同的 200 对（从剩余数据 seed=99 采样，与训练集严格无重叠）
      测试数据：LLVIP 官方 3463 对，全图推理
      优化器：Adam, lr=0.001（来自 DSA-Net）
      调度器：StepLR, step=10, gamma=0.5
      Batch/Patch：8 / 128×128（来自 CDDFuse）
      Epochs：25（来自 DSA-Net）
      梯度裁剪：max_norm=0.01（来自 CDDFuse）
      损失函数：L_pixel + 0.1·L_percep + 0.5·L_grad
    唯一变量：fusion_method
    """
    print("\n" + "="*62)
    print("四组消融实验（唯一变量：融合算子）")
    print("超参数来源：DSA-Net 原文 + CDDFuse 原始代码")
    print("="*62)

    summary = {}
    for i, g in enumerate(ABLATION_GROUPS, 1):
        print(f"\n[{i}/4] {g['desc']}")
        cfg = deepcopy(BASE_CFG)
        cfg["model"]["fusion_method"] = g["fusion_method"]
        cfg["output"]["exp_name"]     = g["exp_name"]
        summary[g["exp_name"]] = {
            "method":  g["fusion_method"],
            "best_AG": run_single(cfg),
        }

    print("\n" + "="*62)
    print("消融汇总（验证集最佳 AG，越高边缘越清晰）")
    print(f"  {'方法':<30} {'AG':>8}")
    print("  " + "-"*40)
    for info in summary.values():
        tag = "  ← 本文方法" if info["method"] == "stp" else ""
        print(f"  {info['method']:<30} {info['best_AG']:>8.4f}{tag}")
    print()
    print("下一步：python evaluate.py  →  测试集完整指标")


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["stp", "bilinear", "nearest", "deconv"])
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="覆盖参数，例如 model.gamma=2 train.epochs=50")
    args = parser.parse_args()

    if not args.method and not args.run_all:
        parser.error("请指定 --method <名称> 或 --run_all")

    cfg = deepcopy(BASE_CFG)
    for ov in args.override:
        k, v = ov.split("=", 1)
        keys = k.split(".")
        node = cfg
        for key in keys[:-1]:
            node = node[key]
        for cast in (int, float, str):
            try: v = cast(v); break
            except ValueError: continue
        node[keys[-1]] = v

    if args.run_all:
        run_all()
    else:
        cfg["model"]["fusion_method"] = args.method
        cfg["output"]["exp_name"]     = f"ablation_{args.method}"
        run_single(cfg)


if __name__ == "__main__":
    main()