"""
scripts/summarize_results.py

汇总所有实验结果，生成可直接复制进论文的 LaTeX 表格。
运行方式：python scripts/summarize_results.py

输出：
  1. 控制台打印：所有方法、所有 γ 值的完整指标
  2. table_main.tex：主对比实验表（γ=4，4 种方法 × 4 个指标）
  3. table_ablation.tex：消融实验表（STP vs Bilinear，3 个 γ 值）
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("./results")
METHODS = ["stp", "bilinear", "nearest", "deconv"]
GAMMAS = [2, 4, 8]
METRICS = ["AG", "EN", "SSIM", "SF"]

METHOD_DISPLAY = {
    "stp":      r"\textbf{STP (Ours)}",
    "bilinear": "Bilinear+Cat",
    "nearest":  "Nearest+Cat",
    "deconv":   "Deconv+Cat",
}


def load_result(method: str, gamma: int) -> dict:
    """从 JSON 文件加载指标结果。"""
    result_file = RESULTS_DIR / f"{method}_gamma{gamma}" / f"test_results_gamma{gamma}.json"
    if not result_file.exists():
        return {}
    with open(result_file) as f:
        return json.load(f)


def find_best(results: dict, metric: str, higher_is_better: bool = True) -> str:
    """在所有方法中找出最佳值，用于在 LaTeX 表格中加粗。"""
    values = {m: results[m].get(metric, float("nan")) for m in results if results[m]}
    if not values:
        return None
    best_method = max(values, key=lambda m: values[m]) if higher_is_better else min(values, key=lambda m: values[m])
    return best_method


def format_cell(value: float, is_best: bool) -> str:
    """格式化表格单元格，最佳值加粗。"""
    if value != value:  # nan
        return "-"
    text = f"{value:.4f}"
    return rf"\textbf{{{text}}}" if is_best else text


def generate_main_table(gamma: int = 4) -> str:
    """生成主对比实验 LaTeX 表格（固定 γ=4）。"""
    results = {m: load_result(m, gamma) for m in METHODS}

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Quantitative comparison at $\gamma={gamma}$ on LLVIP dataset. Best results in \textbf{{bold}}.}}")
    lines.append(r"\label{tab:main_comparison}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & AG$\uparrow$ & EN$\uparrow$ & SSIM$\uparrow$ & SF$\uparrow$ \\")
    lines.append(r"\midrule")

    best_per_metric = {m: find_best(results, m) for m in METRICS}

    for method in METHODS:
        display_name = METHOD_DISPLAY[method]
        row_cells = [display_name]
        for metric in METRICS:
            val = results[method].get(metric, float("nan"))
            is_best = best_per_metric[metric] == method
            row_cells.append(format_cell(val, is_best))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_ablation_table() -> str:
    """生成 γ 消融实验 LaTeX 表格（STP vs Bilinear，3 个 γ 值）。"""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study: AG and EN vs. resolution scale factor $\gamma$.}")
    lines.append(r"\label{tab:gamma_ablation}")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{$\gamma=2$} & \multicolumn{2}{c}{$\gamma=4$} & \multicolumn{2}{c}{$\gamma=8$} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r"Method & AG$\uparrow$ & EN$\uparrow$ & AG$\uparrow$ & EN$\uparrow$ & AG$\uparrow$ & EN$\uparrow$ \\")
    lines.append(r"\midrule")

    for method in ["bilinear", "stp"]:
        display = METHOD_DISPLAY[method]
        cells = [display]
        for g in GAMMAS:
            r = load_result(method, g)
            ag = r.get("AG", float("nan"))
            en = r.get("EN", float("nan"))
            cells.append(f"{ag:.4f}" if ag == ag else "-")
            cells.append(f"{en:.4f}" if en == en else "-")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def print_summary():
    """打印全量结果到控制台。"""
    print("=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    for gamma in GAMMAS:
        print(f"\nγ = {gamma}")
        print(f"{'方法':<15} " + " ".join(f"{m:>8}" for m in METRICS))
        print("-" * 55)
        for method in METHODS:
            r = load_result(method, gamma)
            vals = " ".join(f"{r.get(m, float('nan')):>8.4f}" if r else f"{'N/A':>8}" for m in METRICS)
            print(f"{method:<15} {vals}")


if __name__ == "__main__":
    print_summary()

    # 生成 LaTeX 表格
    output_dir = Path("./paper_figures")
    output_dir.mkdir(exist_ok=True)

    main_table = generate_main_table(gamma=4)
    with open(output_dir / "table_main.tex", "w") as f:
        f.write(main_table)
    print(f"\n主对比表格已保存：{output_dir / 'table_main.tex'}")

    ablation_table = generate_ablation_table()
    with open(output_dir / "table_ablation.tex", "w") as f:
        f.write(ablation_table)
    print(f"消融实验表格已保存：{output_dir / 'table_ablation.tex'}")
    print("\n将 .tex 文件内容复制到论文即可，确保导言区已引入 booktabs 包：")
    print(r"  \usepackage{booktabs}")
