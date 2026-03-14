"""
步骤 1：数据集下载与整理
运行方式：python scripts/01_prepare_data.py --dataset llvip --data_root ./data

LLVIP 数据集说明：
  - 官方地址：https://bupt-ai-cz.github.io/LLVIP/
  - 包含 15488 对严格对齐的红外/可见光图像
  - 训练集：12025 对，测试集：3463 对
  - 图像尺寸：1280×1024（原始），实验中裁剪为 256×256 的 patch

MSRS 数据集说明（可选补充）：
  - 官方地址：https://github.com/Linfeng-Tang/MSRS
  - 道路场景，红外分辨率与可见光一致

本脚本做三件事：
  1. 引导你手动下载数据集（直接下载有权限限制，需要手动）
  2. 验证文件结构是否正确
  3. 生成训练/验证/测试的文件路径列表（.txt 文件）
"""

import os
import argparse
import random
from pathlib import Path


def check_llvip_structure(data_root: str) -> bool:
    """
    检查 LLVIP 数据集目录结构是否正确。
    正确的结构应该是：
    data/LLVIP/
    ├── infrared/
    │   ├── train/  (12025 张 .jpg)
    │   └── test/   (3463 张 .jpg)
    └── visible/
        ├── train/  (12025 张 .jpg)
        └── test/   (3463 张 .jpg)
    """
    required_dirs = [
        "infrared/train", "infrared/test",
        "visible/train", "visible/test"
    ]
    root = Path(data_root) / "LLVIP"
    for d in required_dirs:
        if not (root / d).exists():
            print(f"  [缺失] {root / d}")
            return False
    return True


def generate_file_lists(data_root: str, val_ratio: float = 0.1, seed: int = 42):
    """
    生成训练/验证/测试集的文件路径列表，保存为 .txt 文件。
    格式：每行两个路径（红外路径 可见光路径），空格分隔。
    
    为什么需要这一步：
    - 避免每次训练时重新扫描目录（提高启动速度）
    - 方便控制验证集划分的随机性（固定 seed）
    - 便于在不同数据集上复用同一个 Dataset 类
    """
    root = Path(data_root) / "LLVIP"
    output_dir = Path(data_root) / "file_lists"
    output_dir.mkdir(exist_ok=True)

    for split in ["train", "test"]:
        ir_dir = root / "infrared" / split
        vis_dir = root / "visible" / split

        # 获取所有图像文件名（两个目录的文件名应完全对应）
        ir_files = sorted(ir_dir.glob("*.jpg"))
        pairs = []
        for ir_path in ir_files:
            vis_path = vis_dir / ir_path.name
            if vis_path.exists():
                pairs.append((str(ir_path), str(vis_path)))

        print(f"  {split}: 找到 {len(pairs)} 对图像")

        if split == "train":
            # 从训练集中划分验证集
            random.seed(seed)
            random.shuffle(pairs)
            n_val = int(len(pairs) * val_ratio)
            val_pairs = pairs[:n_val]
            train_pairs = pairs[n_val:]

            # 写入文件
            for name, data in [("train", train_pairs), ("val", val_pairs)]:
                out_path = output_dir / f"llvip_{name}.txt"
                with open(out_path, "w") as f:
                    for ir, vis in data:
                        f.write(f"{ir} {vis}\n")
                print(f"  -> 写入 {out_path}（{len(data)} 对）")
        else:
            out_path = output_dir / f"llvip_test.txt"
            with open(out_path, "w") as f:
                for ir, vis in pairs:
                    f.write(f"{ir} {vis}\n")
            print(f"  -> 写入 {out_path}（{len(pairs)} 对）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="llvip", choices=["llvip", "msrs"])
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()

    print("=" * 50)
    print("数据集准备步骤")
    print("=" * 50)

    print("\n[提示] 请手动下载 LLVIP 数据集：")
    print("  1. 访问 https://bupt-ai-cz.github.io/LLVIP/")
    print("  2. 下载 infrared.zip 和 visible.zip")
    print(f"  3. 解压到 {args.data_root}/LLVIP/ 目录下")
    print("  4. 确保目录结构如下：")
    print("     data/LLVIP/infrared/train/, data/LLVIP/infrared/test/")
    print("     data/LLVIP/visible/train/,  data/LLVIP/visible/test/")
    print()

    print("\n[检查] 验证目录结构...")
    if not check_llvip_structure(args.data_root):
        print("\n[错误] 数据集目录结构不完整，请按照上述提示下载并整理数据集。")
        print("验证通过后重新运行本脚本。")
        return

    print("  目录结构正确！")
    print("\n[生成] 创建文件路径列表...")
    generate_file_lists(args.data_root)
    print("\n数据准备完成！")


if __name__ == "__main__":
    main()
