#!/usr/bin/env python3
"""
Figure 3 — Risk prioritization among FDA-uncategorized drugs.

布局：双栏宽 183 mm，2×2 网格
  A（左上）：vLess vs Ambiguous 小提琴图 + 内嵌箱线图（证明整体分布无差异）
  B（右上）：659 药物风险分数直方图，vLess/Ambiguous 堆叠色 + Top5% 阈值线
  C（左下）：尾部富集森林图（Top5%/10%/20% 的 OR + Fisher P）
  D（右下）：7 个验证药在 659 排序中的位置（横向散点 + Top5% 阈值线）
"""

import sys
sys.path.insert(0, r"D:/abraca")
import plot_utils as pu
pu.apply_paper_style()

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 数据加载（v2 重算产物）
# =============================================================================
US = json.load(open(r"D:/DILI/data/unknown_summary_v2.json"))
rank_df = pd.read_excel(r"D:/DILI/data/DILI_unknown_prediction_rank_v2.xlsx")
tail_df = pd.read_excel(r"D:/DILI/data/DILI_tail_enrichment_analysis_v2.xlsx")

vless = rank_df[rank_df["DILI_category"] == "vLess-DILI-concern"]["DILI_risk_score"].values
ambig = rank_df[rank_df["DILI_category"] == "Ambiguous-DILI-concern"]["DILI_risk_score"].values
all_scores = rank_df["DILI_risk_score"].values
P_MW = US["mannwhitney_p"]

# 验证药（分数由 v2 模型重算；Diclofenac 池外无分数）
_score = {d["drug"]: (d["score"], d["rank"]) for d in US["validation_drugs"]}
validation_drugs = [
    ("Chlorpromazine", _score["Chlorpromazine"][0], "positive_control"),
    ("Sulfamethizole", _score["Sulfamethizole"][0], "high_risk_candidate"),
    ("Cabotegravir",   _score["Cabotegravir"][0],   "high_risk_candidate"),
    ("Mebendazole",    _score["Mebendazole"][0],    "high_risk_candidate"),
    ("Oxybutynin",     _score["Oxybutynin"][0],     "negative_control"),
    ("Tolterodine",    _score["Tolterodine"][0],    "negative_control"),
    ("Metformin",      _score["Metformin"][0],      "negative_control"),
]
print("验证药分数:", [(n, round(s, 4)) for n, s, _ in validation_drugs])

# Top5% 阈值 = 第 32 名分数（tail 分析用 floor(659*0.05)=32）
top5_threshold = float(np.sort(all_scores)[::-1][31])
print("Top5%% 阈值 = %.4f (边界药 %s)" % (
    top5_threshold, rank_df.sort_values("DILI_risk_score", ascending=False).iloc[31]["CompoundName"]))

# =============================================================================
# 画布 2×2
# =============================================================================
fig = plt.figure(figsize=pu.figsize(pu.DOUBLE_COL_MM, height_mm=118))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                      width_ratios=[1.0, 1.0], hspace=0.55, wspace=0.45,
                      top=0.90, bottom=0.12)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

# =============================================================================
# A：小提琴图 + 内嵌箱线图
# =============================================================================
cat_colors = [pu.CATEGORY_COLORS["vLess-DILI-concern"],
              pu.CATEGORY_COLORS["Ambiguous-DILI-concern"]]
data_A = [vless, ambig]
labels_A = ["vLess\n(n=328)", "Ambiguous\n(n=331)"]

parts = axA.violinplot(data_A, positions=[0, 1], showmeans=False,
                       showmedians=False, showextrema=False)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(cat_colors[i])
    pc.set_alpha(0.45)
    pc.set_edgecolor("none")
# 内嵌箱线图
bp = axA.boxplot(data_A, positions=[0, 1], widths=0.12, patch_artist=True,
                 showfliers=False,
                 medianprops=dict(color="black", linewidth=1.0),
                 boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
                 whiskerprops=dict(color="black", linewidth=0.8),
                 capprops=dict(color="black", linewidth=0.8))
axA.set_xticks([0, 1])
axA.set_xticklabels(labels_A, fontsize=6.5)
axA.set_ylabel("DILI risk score")
axA.set_ylim(0, 1)
axA.tick_params(axis="x", labelsize=6.5)
axA.tick_params(axis="y", labelsize=6)
# 显著性标注
pu.add_significance(axA, 0, 1, 0.95, "P = {:.3f}\n(Mann-Whitney U)".format(P_MW))
pu.label_panel(axA, "A")

# =============================================================================
# B：风险分数直方图（堆叠色）
# =============================================================================
bins = np.arange(0.0, 0.9, 0.04)
axB.hist([vless, ambig], bins=bins, stacked=True,
         color=cat_colors, alpha=0.75, edgecolor="white", linewidth=0.3,
         label=["vLess", "Ambiguous"])
axB.axvline(top5_threshold, color=pu.OKABE_ITO["vermillion"],
            linewidth=1.2, linestyle="--")
axB.text(top5_threshold + 0.01, axB.get_ylim()[1] * 0.9,
         "Top 5%", fontsize=6, color=pu.OKABE_ITO["vermillion"],
         ha="left", fontweight="bold")
axB.set_xlabel("DILI risk score")
axB.set_ylabel("Number of compounds")
axB.tick_params(axis="x", labelsize=6)
axB.tick_params(axis="y", labelsize=6)
axB.legend(fontsize=6, frameon=False, loc="upper left")
pu.label_panel(axB, "B")

# =============================================================================
# C：尾部富集森林图（Top5%/10%/20% 的 OR）
# =============================================================================
labels_C = tail_df["Top_fraction"].tolist()
ors_C = tail_df["Odds_ratio"].values
ps_C = tail_df["Fisher_p"].values
ypos_C = np.arange(len(labels_C))[::-1]

# 颜色：Top5% 突出（vermillion），其余灰
colors_C = [pu.OKABE_ITO["vermillion"] if i == 0 else "#888780"
            for i in range(len(labels_C))]
axC.scatter(ors_C, ypos_C, s=60, color=colors_C, zorder=3)
lo_C = tail_df["OR_CI_low"].values
hi_C = tail_df["OR_CI_high"].values
axC.errorbar(ors_C, ypos_C, xerr=[np.clip(ors_C - lo_C, 0, None), np.clip(hi_C - ors_C, 0, None)],
             fmt="none", ecolor="#888780", elinewidth=0.8, capsize=2, zorder=2)
axC.axvline(1, color="#888780", linewidth=0.8, linestyle="--")
axC.set_yticks(ypos_C)
axC.set_yticklabels(labels_C, fontsize=6.5)
axC.set_xlabel("Odds ratio (vLess enrichment)")
axC.set_xscale("log")
axC.set_xlim(0.7, 7.5)
axC.tick_params(axis="x", labelsize=6)
# 标注 OR 和 P 值
for yi, or_, p in zip(ypos_C, ors_C, ps_C):
    axC.text(or_ * 1.10, yi + 0.20, f"OR={or_:.2f}, P={p:.3f}",
             fontsize=5.2, ha="left", va="bottom",
             color=pu.OKABE_ITO["vermillion"] if or_ > 3 else "#444441",
             fontweight="bold" if or_ > 3 else "normal")
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)
pu.label_panel(axC, "C")

# =============================================================================
# D：验证药在 659 排序中的位置（横向散点）
# =============================================================================
# 验证药（按分数从上到下排）
drugs_sorted = sorted(validation_drugs, key=lambda x: -x[1])
y_positions = np.arange(len(drugs_sorted))
for i, (name, score, group) in enumerate(drugs_sorted):
    color = pu.DRUG_GROUP_COLORS[group]
    marker = pu.DRUG_GROUP_MARKERS[group]
    axD.scatter([score], [i], s=60, color=color, marker=marker,
                edgecolor="black", linewidth=0.5, zorder=3)
    axD.text(score + 0.035, i, name, fontsize=6, va="center", ha="left")

axD.axvline(top5_threshold, color=pu.OKABE_ITO["vermillion"],
            linewidth=1.0, linestyle="--")
axD.text(top5_threshold + 0.01, len(drugs_sorted) - 0.5, "Top 5%",
         fontsize=5.5, color=pu.OKABE_ITO["vermillion"], ha="left")

axD.set_yticks([])
axD.set_ylim(-0.8, len(drugs_sorted) - 0.2)
axD.set_xlabel("DILI risk score", fontsize=6)
axD.set_xlim(0.2, 1.05)
axD.tick_params(axis="x", labelsize=6)
axD.spines["top"].set_visible(False)
axD.spines["right"].set_visible(False)
axD.spines["left"].set_visible(False)
# 图例
from matplotlib.lines import Line2D
legend_D = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=pu.DRUG_GROUP_COLORS["positive_control"],
           markersize=6, label="Positive control"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=pu.DRUG_GROUP_COLORS["high_risk_candidate"],
           markersize=6, label="High-risk candidate"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=pu.DRUG_GROUP_COLORS["negative_control"],
           markersize=6, label="Negative control"),
]
axD.legend(handles=legend_D, fontsize=5.5, frameon=False, loc="lower left")
pu.label_panel(axD, "D")

# Diclofenac 说明（放在 D panel 正下方，两行，在 "is" 处断句）
posD = axD.get_position()
fig.text(0.5 * (posD.x0 + posD.x1), posD.y0 - 0.085,
         "Diclofenac (out-of-pool positive control,\n"
         "Most-DILI-concern) is not scored by the model",
         fontsize=5.5, ha="center", va="top", color="#5F5E5A",
         style="italic", linespacing=1.4)

# =============================================================================
# 导出
# =============================================================================
pu.save_figure(fig, r"D:/DILI/result/new_figure/Figure3",
               formats=("pdf", "png"), dpi=600, tight=False)
plt.close(fig)
print("Figure3 完成")
print("Top5% threshold = {:.3f}".format(top5_threshold))
