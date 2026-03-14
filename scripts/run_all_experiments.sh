#!/bin/bash
# ============================================================
# scripts/run_all_experiments.sh
#
# 一键运行所有实验：4 种方法 × 3 个 γ 值 = 12 组训练任务
# 如果只有单卡 GPU，建议按顺序跑；如果有多卡，可以并行。
#
# 运行方式：bash scripts/run_all_experiments.sh
#
# 预计时间（单张 RTX 3090，200 epoch，LLVIP）：
#   每组约 3–4 小时，12 组共约 36–48 小时
#   建议用 screen 或 tmux 跑后台任务
# ============================================================

set -e  # 任何命令失败就停止

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate stp_fusion

# 检查 GPU 是否可用
python -c "import torch; assert torch.cuda.is_available(), 'No GPU found!'"
echo "GPU 检测通过，开始实验..."

# ── 实验参数定义 ──────────────────────────────────
METHODS=("stp" "bilinear" "nearest" "deconv")
GAMMAS=(2 4 8)
CONFIG="configs/default.yaml"

# ── 主实验：γ=4（论文主表格）──────────────────────
echo ""
echo "===== 主实验：γ=4 ====="
for METHOD in "${METHODS[@]}"; do
    EXP_NAME="${METHOD}_gamma4"
    echo ""
    echo ">>> 训练：${EXP_NAME}"
    python train.py \
        --config ${CONFIG} \
        --override \
            model.fusion_method=${METHOD} \
            model.gamma=4 \
            output.exp_name=${EXP_NAME}
    
    echo ">>> 评估：${EXP_NAME}"
    python evaluate.py \
        --checkpoint results/${EXP_NAME}/best_model.pth \
        --gamma 4
done

# ── 消融实验：不同 γ 值（论文消融表格）──────────────
echo ""
echo "===== 消融实验：不同 γ 值 ====="
for GAMMA in "${GAMMAS[@]}"; do
    for METHOD in stp bilinear; do  # 简化：只跑 STP 和最强 baseline
        EXP_NAME="${METHOD}_gamma${GAMMA}"
        if [ -f "results/${EXP_NAME}/best_model.pth" ]; then
            echo ">>> 已存在，跳过训练：${EXP_NAME}"
        else
            echo ""
            echo ">>> 训练：${EXP_NAME}"
            python train.py \
                --config ${CONFIG} \
                --override \
                    model.fusion_method=${METHOD} \
                    model.gamma=${GAMMA} \
                    output.exp_name=${EXP_NAME}
        fi
        
        echo ">>> 评估：${EXP_NAME}"
        python evaluate.py \
            --checkpoint results/${EXP_NAME}/best_model.pth \
            --gamma ${GAMMA}
    done
done

echo ""
echo "===== 所有实验完成！====="
echo ""
echo "下一步：汇总结果到论文表格"
echo "  python scripts/summarize_results.py"
echo ""
echo "生成论文可视化图表（请先手动选好 ROI 坐标）："
echo "  python visualize.py \\"
echo "    --checkpoints results/stp_gamma4/best_model.pth results/bilinear_gamma4/best_model.pth \\"
echo "    --names 'STP (Ours)' 'Bilinear' \\"
echo "    --sample_idx 5 20 42 \\"
echo "    --roi 100 200 180 340"
