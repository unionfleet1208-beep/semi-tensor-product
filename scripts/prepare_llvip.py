"""
prepare_llvip.py

LLVIP 数据集前期处理一键脚本。完成以下工作：
  1. 验证目录结构完整性
  2. 检查红外/可见光图像对是否一一对应
  3. 从训练集随机采样 200 对作为训练集（seed=42）
  4. 从剩余图像中采样 200 对作为验证集（seed=99），严格无重叠
  5. 生成测试集列表（官方全部测试图像）
  6. 复制对应的 XML 标注文件到各子集目录（如果存在）
  7. 打印数据统计摘要

运行方式：
  python scripts/prepare_llvip.py --data_root ./data/LLVIP
  python scripts/prepare_llvip.py --data_root ./data/LLVIP --train_n 200 --val_n 200
"""

import os
import random
import shutil
import argparse
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# 目录结构验证
# ──────────────────────────────────────────────────────────────

REQUIRED_DIRS = [
    "infrared/train",
    "infrared/test",
    "visible/train",
    "visible/test",
]

def verify_structure(root: Path) -> bool:
    ok = True
    for d in REQUIRED_DIRS:
        p = root / d
        if not p.exists():
            print(f"  [缺失] {p}")
            ok = False
        else:
            n = len(list(p.glob("*.jpg"))) + len(list(p.glob("*.png")))
            print(f"  [OK]   {p}  ({n} 张图)")
    return ok


# ──────────────────────────────────────────────────────────────
# 图像对配准检查
# ──────────────────────────────────────────────────────────────

def get_paired_files(root: Path, split: str):
    """
    返回 (ir_path, vis_path) 列表，只保留两个模态都存在的图像对。
    LLVIP 的红外和可见光文件名完全一致，直接按文件名匹配。
    """
    ir_dir  = root / "infrared" / split
    vis_dir = root / "visible"  / split

    ir_files  = {f.name: f for f in ir_dir.glob("*.jpg")}
    vis_files = {f.name: f for f in vis_dir.glob("*.jpg")}

    # 取交集（两个模态都存在的文件名）
    common = sorted(set(ir_files.keys()) & set(vis_files.keys()))
    pairs  = [(str(ir_files[n]), str(vis_files[n])) for n in common]

    # 报告缺失情况
    only_ir  = set(ir_files)  - set(vis_files)
    only_vis = set(vis_files) - set(ir_files)
    if only_ir:
        print(f"  [警告] {split}: {len(only_ir)} 张红外无对应可见光，已跳过")
    if only_vis:
        print(f"  [警告] {split}: {len(only_vis)} 张可见光无对应红外，已跳过")

    return pairs


# ──────────────────────────────────────────────────────────────
# XML 标注文件处理
# ──────────────────────────────────────────────────────────────

def copy_annotations(root: Path, pairs, out_xml_dir: Path):
    """
    LLVIP 的 Annotations 目录包含 VOC 格式的 XML 标注（行人检测用）。
    把选中图像对应的 XML 复制到指定目录，方便后续下游任务使用。
    如果 Annotations 目录不存在，直接跳过（不影响融合训练）。
    """
    ann_dir = root / "Annotations"
    if not ann_dir.exists():
        return 0

    out_xml_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for ir_path, _ in pairs:
        stem   = Path(ir_path).stem          # 文件名去掉扩展名
        xml_src = ann_dir / f"{stem}.xml"
        if xml_src.exists():
            shutil.copy2(xml_src, out_xml_dir / xml_src.name)
            copied += 1

    return copied


# ──────────────────────────────────────────────────────────────
# 文件列表写入
# ──────────────────────────────────────────────────────────────

def write_list(pairs, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ir_p, vis_p in pairs:
            f.write(f"{ir_p} {vis_p}\n")
    print(f"  → {out_path}  ({len(pairs)} 对)")


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLVIP 数据集前期处理")
    parser.add_argument("--data_root", default="./data/LLVIP",
                        help="LLVIP 根目录（包含 infrared/ 和 visible/）")
    parser.add_argument("--output_dir", default="./data/file_lists",
                        help="文件列表输出目录")
    parser.add_argument("--train_n", type=int, default=200,
                        help="训练集采样对数（默认 200，领域标准）")
    parser.add_argument("--val_n",   type=int, default=200,
                        help="验证集采样对数（默认 200）")
    parser.add_argument("--seed_train", type=int, default=42)
    parser.add_argument("--seed_val",   type=int, default=99)
    parser.add_argument("--copy_xml",   action="store_true",
                        help="是否复制对应的 XML 标注文件")
    args = parser.parse_args()

    root       = Path(args.data_root)
    output_dir = Path(args.output_dir)

    print("=" * 55)
    print("LLVIP 数据准备")
    print("=" * 55)

    # ── Step 1：验证目录结构 ──────────────────────────────────
    print("\n[Step 1] 验证目录结构")
    if not verify_structure(root):
        print("\n[错误] 目录结构不完整，请检查数据集路径。")
        return

    # ── Step 2：获取训练集所有图像对 ─────────────────────────
    print("\n[Step 2] 扫描图像对")
    train_all = get_paired_files(root, "train")
    test_all  = get_paired_files(root, "test")
    print(f"  训练集总对数：{len(train_all)}")
    print(f"  测试集总对数：{len(test_all)}")

    if len(train_all) < args.train_n + args.val_n:
        print(f"[错误] 训练集图像对不足 {args.train_n + args.val_n} 对，"
              f"实际只有 {len(train_all)} 对")
        return

    # ── Step 3：划分训练 / 验证集（严格无重叠）──────────────
    print(f"\n[Step 3] 划分训练/验证集（无重叠）")

    random.seed(args.seed_train)
    train_pairs = random.sample(train_all, args.train_n)
    train_set   = set(map(lambda x: x[0], train_pairs))  # 用 ir_path 作为唯一键

    remaining   = [p for p in train_all if p[0] not in train_set]
    random.seed(args.seed_val)
    val_pairs   = random.sample(remaining, args.val_n)

    print(f"  训练集：{len(train_pairs)} 对  (seed={args.seed_train})")
    print(f"  验证集：{len(val_pairs)} 对  (seed={args.seed_val})")
    print(f"  剩余未使用：{len(remaining) - args.val_n} 对")

    # 双重确认无重叠
    train_names = {Path(p[0]).stem for p in train_pairs}
    val_names   = {Path(p[0]).stem for p in val_pairs}
    overlap     = train_names & val_names
    if overlap:
        print(f"  [错误] 发现 {len(overlap)} 对重叠，请检查随机种子设置！")
        return
    print(f"  [验证] 训练集与验证集无重叠 ✓")

    # ── Step 4：写入文件列表 ──────────────────────────────────
    print(f"\n[Step 4] 写入文件列表 → {output_dir}")
    write_list(train_pairs, output_dir / "llvip_train_200.txt")
    write_list(val_pairs,   output_dir / "llvip_val_200.txt")
    write_list(test_all,    output_dir / "llvip_test.txt")
    # 同时写入完整训练集列表（供需要时使用）
    write_list(train_all,   output_dir / "llvip_train_full.txt")

    # ── Step 5：复制 XML 标注（可选）────────────────────────
    if args.copy_xml:
        print(f"\n[Step 5] 复制 XML 标注文件")
        for name, pairs in [("train", train_pairs),
                             ("val",   val_pairs),
                             ("test",  test_all)]:
            xml_out = output_dir / "annotations" / name
            n = copy_annotations(root, pairs, xml_out)
            if n > 0:
                print(f"  {name}: {n} 个 XML → {xml_out}")
            else:
                print(f"  {name}: 未找到 XML 文件（不影响融合训练，仅下游检测任务需要）")
    else:
        print(f"\n[Step 5] 跳过 XML 复制（需要时加 --copy_xml）")

    # ── Step 6：统计摘要 ──────────────────────────────────────
    print(f"\n{'='*55}")
    print("数据准备完成！统计摘要：")
    print(f"  训练集：{len(train_pairs)} 对  →  llvip_train_200.txt")
    print(f"  验证集：{len(val_pairs)} 对  →  llvip_val_200.txt")
    print(f"  测试集：{len(test_all)} 对  →  llvip_test.txt")
    print(f"  完整训练集：{len(train_all)} 对  →  llvip_train_full.txt")
    print()
    print("下一步，直接运行训练：")
    print("  python train.py --run_all           # 四组消融实验")
    print("  python train.py --method stp        # 仅训练 STP 方法")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()