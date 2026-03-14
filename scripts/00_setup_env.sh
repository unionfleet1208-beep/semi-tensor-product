#!/bin/bash
# ============================================================
# 步骤 0：环境配置
# 运行方式：bash scripts/00_setup_env.sh
# 建议在 conda 虚拟环境中运行，Python >= 3.9
# ============================================================

echo ">>> 创建 conda 环境（如已存在会跳过）"
conda create -n stp_fusion python=3.10 -y
conda activate stp_fusion

echo ">>> 安装 PyTorch（CUDA 11.8，按实际 CUDA 版本修改）"
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

echo ">>> 安装其他依赖"
pip install \
    numpy==1.24.3 \
    opencv-python==4.8.0.76 \
    Pillow==10.0.0 \
    scikit-image==0.21.0 \
    scipy==1.11.0 \
    matplotlib==3.7.2 \
    tqdm==4.65.0 \
    pyyaml==6.0.1 \
    tensorboard==2.14.0 \
    piq==0.8.0        # 计算 VIF 指标

echo ">>> 验证安装"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
echo "环境配置完成！"
