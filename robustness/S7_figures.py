# -*- coding: utf-8 -*-
"""
Step 7 — 图表 (matplotlib)
产出: figures/FigS1_agreement_heatmap.png
      figures/FigS2_descriptor_delta.png
      figures/FigS3_consensus_layers.png
      figures/FigS4_label_sensitivity.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
FIG = dc.FIG; FIG.mkdir(parents=True, exist_ok=True)

log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

P("=" * 72); P("Step 7 — 图表"); P("=" * 72)

# ======================================================================
# FigS1: 模型两两 Spearman 热图
# ======================================================================
sp = pd.read_csv(dc.OUT / "S5_pairwise_spearman.csv", index_col=0)
fig, ax = plt.subplots(figsize=(7.5, 6.2))
im = ax.imshow(sp.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(sp))); ax.set_xticklabels(sp.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(sp))); ax.set_yticklabels(sp.index, fontsize=8)
for i in range(len(sp)):
    for j in range(len(sp)):
        ax.text(j, i, f"{sp.values[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(sp.values[i, j]) > 0.6 else "black", fontsize=7)
ax.set_title("Fig S1. 跨模型风险排序一致性 (Spearman ρ, 659 待评药)", fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
fig.tight_layout(); fig.savefig(FIG / "FigS1_agreement_heatmap.png", dpi=200); plt.close()
P("[7.1] FigS1 -> figures/FigS1_agreement_heatmap.png")

# ======================================================================
# FigS2: 描述符空间 ΔAUC 条形图
# ======================================================================
cmp_df = pd.read_csv(dc.OUT / "S4_descriptor_comparison.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.5))
colors = ["#c0392b" if "3D" in s else "#2c7fb8" for s in cmp_df["feature_set"]]
bars = ax.barh(cmp_df["feature_set"], cmp_df["auc_mean"], color=colors, alpha=0.85)
ax.errorbar(cmp_df["auc_mean"], range(len(cmp_df)),
            xerr=[cmp_df["auc_mean"] - cmp_df["ci_lo"], cmp_df["ci_hi"] - cmp_df["auc_mean"]],
            fmt="none", ecolor="black", capsize=4, lw=1.2)
ax.axvline(0.8046, ls="--", color="gray", lw=1, label="F3 基线 0.8046")
ax.set_xlabel("5-fold CV AUC (L1, RF)"); ax.set_xlim(0.55, 0.90)
ax.set_title("Fig S2. 描述符空间比较：3D 未带来增益 (ΔAUC = −0.0145)", fontsize=11)
ax.legend(fontsize=8)
for b, v in zip(bars, cmp_df["auc_mean"]):
    ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "FigS2_descriptor_delta.png", dpi=200); plt.close()
P("[7.2] FigS2 -> figures/FigS2_descriptor_delta.png")

# ======================================================================
# FigS3: 共识分层
# ======================================================================
lay = pd.read_csv(dc.OUT / "S5_consensus_layers.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axes[0]
cols = {"Consensus_HIGH": "#c0392b", "Consensus_LOW": "#2c7fb8", "Discordant": "#f39c12"}
ax.bar(lay["layer"], lay["n"], color=[cols.get(x, "gray") for x in lay["layer"]])
for i, (_, r) in enumerate(lay.iterrows()):
    ax.text(i, r["n"] + 8, str(r["n"]), ha="center", fontsize=9)
ax.set_ylabel("药物数"); ax.set_title("(a) 共识分层规模 (Top-10%, 7 模型)", fontsize=10)
ax = axes[1]
x = np.arange(len(lay)); w = 0.38
ax.bar(x - w/2, lay["n_vLess"], w, label="vLess", color="#8e44ad", alpha=0.85)
ax.bar(x + w/2, lay["n_amb"], w, label="Ambiguous", color="#16a085", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(lay["layer"], rotation=15, fontsize=8)
ax.set_ylabel("药物数"); ax.set_title("(b) 各层类别构成", fontsize=10); ax.legend(fontsize=8)
fig.suptitle("Fig S3. 多模型共识分层", fontsize=12)
fig.tight_layout(); fig.savefig(FIG / "FigS3_consensus_layers.png", dpi=200); plt.close()
P("[7.3] FigS3 -> figures/FigS3_consensus_layers.png")

# ======================================================================
# FigS4: 标签敏感性
# ======================================================================
lab = pd.read_csv(dc.OUT / "S3_label_sensitivity.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.3))
plot = lab.dropna(subset=["auc_mean"])
colors4 = ["#2c7fb8" if s in ("L1",) else "#7fb3d5" if s.startswith("L") else "#f39c12"
           for s in plot["scheme"]]
ax.bar(plot["scheme"], plot["auc_mean"], color=colors4, alpha=0.9)
ax.errorbar(range(len(plot)), plot["auc_mean"],
            yerr=[plot["auc_mean"] - plot["ci_lo"].fillna(plot["auc_mean"]),
                  plot["ci_hi"].fillna(plot["auc_mean"]) - plot["auc_mean"]],
            fmt="none", ecolor="black", capsize=5, lw=1.2)
for i, (_, r) in enumerate(plot.iterrows()):
    ax.text(i, r["auc_mean"] + 0.008, f"{r['auc_mean']:.4f}", ha="center", fontsize=8)
ax.set_ylabel("CV AUC (Most vs No)"); ax.set_ylim(0.7, 0.87)
ax.set_title("Fig S4. 标签编码方案敏感性 (5-fold CV)", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "FigS4_label_sensitivity.png", dpi=200); plt.close()
P("[7.4] FigS4 -> figures/FigS4_label_sensitivity.png")

# ======================================================================
# FigS5: Ambiguous 定位 (L4 副产品)
# ======================================================================
pl = pd.read_csv(dc.OUT / "S3_ambiguous_placement.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.3))
ax.hist(pl["P_Most"], bins=40, color="#c0392b", alpha=0.7, label="P(Most)")
ax.hist(pl["P_Less"], bins=40, color="#f39c12", alpha=0.6, label="P(Less)")
ax.hist(pl["P_No"], bins=40, color="#2c7fb8", alpha=0.6, label="P(No)")
ax.axvline(pl["P_Less"].mean(), ls="--", color="#f39c12", lw=1.5)
ax.axvline(pl["P_No"].mean(), ls="--", color="#2c7fb8", lw=1.5)
ax.set_xlabel("OOF 预测概率"); ax.set_ylabel("Ambiguous 药数量")
ax.set_title(f"Fig S5. 331 个 Ambiguous 药的模型定位\n"
             f"（均值 P(Less)={pl['P_Less'].mean():.3f} vs P(No)={pl['P_No'].mean():.3f}，"
             f"argmax 分布 {pl['argmax'].value_counts().to_dict()}）", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "FigS5_ambiguous_placement.png", dpi=200); plt.close()
P("[7.5] FigS5 -> figures/FigS5_ambiguous_placement.png")

(dc.OUT / "S7_step7.log").write_text("\n".join(log), encoding="utf-8")
P("\nStep 7 完成。")
