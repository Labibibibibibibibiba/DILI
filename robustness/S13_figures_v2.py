# -*- coding: utf-8 -*-
"""
S13 — Supplementary figures v2 (redrawn in the "药1" (DILI main-paper) style)

Reference style: D:\\DILI\\paper\\figures\\Figure1-3
  - base: D:\\abraca\\plot_utils.py  (Okabe-Ito colour-blind-safe palette,
    Arial 8pt, no top/right spines, panel labels A/B/C/D, 300+ dpi, PDF+PNG)
  - English labels, concise, no CJK in the figure canvas.

Outputs (figures_v2/):
  FigS1_rank_agreement.pdf/.png      跨模型排序一致性 (Spearman + Jaccard)
  FigS2_descriptor_space.pdf/.png     描述符空间比较 (F1-F5, 3D 无增益)
  FigS3_consensus_layers.pdf/.png     共识分层 (规模 + 类别构成 + 票数分布)
  FigS4_label_sensitivity.pdf/.png    标签编码敏感性 (L1-L4 AUROC)
  FigS5_ambiguous_placement.pdf/.png  331 个 Ambiguous 药的定位
  FigS6_rank_stability.pdf/.png       36 组合下 8 药排名区间 (S11)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

import sys
sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc
import plot_utils as pu

pu.apply_paper_style()

OUT = dc.OUT
FIG = dc.DILI1 / "figures_v2"
FIG.mkdir(parents=True, exist_ok=True)

OI = pu.OKABE_ITO
log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

P("=" * 72); P("S13 — Supplementary figures v2 (药1 style)"); P("=" * 72)


# ======================================================================
# FigS1 — cross-model rank agreement (Spearman + Jaccard Top-10%)
# ======================================================================
sp = pd.read_csv(OUT / "S5_pairwise_spearman.csv", index_col=0)
jac = pd.read_csv(OUT / "S5_jaccard_top10.csv", index_col=0)

SHORT = {
    "L1_F1_A1": "RF/Morgan",
    "L1_F3_A1": "RF/M+2D",
    "L1_F2_A1": "RF/2D",
    "L1_F3_A2": "XGB/M+2D",
    "L1_F3_A3": "LR/M+2D",
    "L1_F4_A1": "RF/3D",
    "L1_F5_A1": "RF/M+2D+3D",
}
lbl = [SHORT[c] for c in sp.columns]

fig, axes = plt.subplots(1, 2, figsize=pu.figsize(pu.DOUBLE_COL_MM, height_mm=92),
                         layout="constrained")

ax = axes[0]
im = ax.imshow(sp.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
ax.set_xticks(range(len(lbl))); ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=6.5)
ax.set_yticks(range(len(lbl))); ax.set_yticklabels(lbl, fontsize=6.5)
for i in range(len(sp)):
    for j in range(len(sp)):
        v = sp.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                color="white" if abs(v) > 0.55 else "black")
ax.set_title("Spearman rank correlation", fontsize=8)
cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
cb.set_label("Spearman $\\rho$", fontsize=7); cb.ax.tick_params(labelsize=6)
pu.label_panel(ax, "A")

ax = axes[1]
im = ax.imshow(jac.values, cmap="Blues", vmin=0, vmax=1, aspect="equal")
ax.set_xticks(range(len(lbl))); ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=6.5)
ax.set_yticks(range(len(lbl))); ax.set_yticklabels(lbl, fontsize=6.5)
for i in range(len(jac)):
    for j in range(len(jac)):
        v = jac.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                color="white" if v > 0.55 else "black")
ax.set_title("Top-10% set overlap (Jaccard)", fontsize=8)
cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
cb.set_label("Jaccard index", fontsize=7); cb.ax.tick_params(labelsize=6)
pu.label_panel(ax, "B")

fig.suptitle("Figure S1. Cross-model rank agreement on the 659 uncategorised compounds",
             fontsize=9)
pu.save_figure(fig, FIG / "FigS1_rank_agreement", formats=("pdf", "png"), dpi=600)
plt.close(fig)
P("[S1] FigS1_rank_agreement")


# ======================================================================
# FigS2 — descriptor-space comparison (F1-F5)
# ======================================================================
cmp_df = pd.read_csv(OUT / "S4_descriptor_comparison.csv")
NICE = {
    "F1_Morgan":   "F1\nMorgan\n(1024)",
    "F2_2Ddesc":   "F2\n2D desc\n(7)",
    "F3_Morgan+2D": "F3\nMorgan\n+2D (1031)",
    "F4_3Donly":   "F4\n3D only\n(11)",
    "F5_2D+3D":    "F5\nMorgan\n+2D+3D",
}
order = ["F1_Morgan", "F2_2Ddesc", "F3_Morgan+2D", "F4_3Donly", "F5_2D+3D"]
cmp_df = cmp_df.set_index("feature_set").loc[order].reset_index()

fig, ax = plt.subplots(figsize=pu.figsize(pu.DOUBLE_COL_MM, height_mm=76),
                       layout="constrained")
x = np.arange(len(cmp_df))
# 2D spaces = blue family; 3D-containing = orange/amber family (redundant cue via hatch)
cols = [OI["blue"] if "3D" not in s else OI["orange"] for s in cmp_df["feature_set"]]
hatches = ["" if "3D" not in s else "//" for s in cmp_df["feature_set"]]
bars = ax.bar(x, cmp_df["auc_mean"], color=cols, edgecolor="black",
              linewidth=0.5, width=0.62)
for b, h in zip(bars, hatches):
    b.set_hatch(h)
ax.errorbar(x, cmp_df["auc_mean"],
            yerr=[cmp_df["auc_mean"] - cmp_df["ci_lo"],
                  cmp_df["ci_hi"] - cmp_df["auc_mean"]],
            fmt="none", ecolor="black", capsize=3, lw=1.0)
ax.axhline(0.8046, ls="--", color="gray", lw=0.9)
ax.text(0.34, 0.8046 + 0.006, "F3 baseline 0.805", ha="left",
        va="bottom", fontsize=6.2, color="gray")
for b, v, hi in zip(bars, cmp_df["auc_mean"], cmp_df["ci_hi"]):
    ax.text(b.get_x() + b.get_width() / 2, hi + 0.006, f"{v:.3f}",
            ha="center", va="bottom", fontsize=6.2)
ax.set_xticks(x); ax.set_xticklabels([NICE[s] for s in cmp_df["feature_set"]], fontsize=6.6)
ax.set_ylabel("5-fold CV AUROC (L1, RF)")
ax.set_ylim(0.55, 0.94)
ax.set_title("Figure S2. Descriptor-space comparison:\nthree-dimensional features give no gain",
             fontsize=9.0)
pu.label_panel(ax, "A", x=-0.10)
ax.text(0.02, 0.97,
        "$\\Delta$AUROC (F5 $-$ F3) = $-$0.015\n95% CI [$-$0.024, $-$0.005]",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.4,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.5))
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=OI["blue"], edgecolor="black", label="2D space"),
                   Patch(facecolor=OI["orange"], edgecolor="black", hatch="//",
                         label="contains 3D")],
          fontsize=6.4, loc="upper right")
pu.save_figure(fig, FIG / "FigS2_descriptor_space", formats=("pdf", "png"), dpi=600)
plt.close(fig)
P("[S2] FigS2_descriptor_space")


# ======================================================================
# FigS3 — consensus layering (size + composition + vote histogram)
# ======================================================================
lay = pd.read_csv(OUT / "S5_consensus_layers.csv")
sets = pd.read_csv(OUT / "S5_high_risk_sets.csv")

fig, axes = plt.subplots(1, 3, figsize=pu.figsize(pu.DOUBLE_COL_MM, height_mm=78),
                         layout="constrained",
                         gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})

# (A) layer sizes
ax = axes[0]
lorder = ["Consensus_HIGH", "Discordant", "Consensus_LOW"]
lnice = {"Consensus_HIGH": "Consensus\nHIGH", "Discordant": "Discordant",
         "Consensus_LOW": "Consensus\nLOW"}
lcol = {"Consensus_HIGH": OI["vermillion"], "Discordant": OI["orange"],
        "Consensus_LOW": OI["blue"]}
lay_i = lay.set_index("layer").loc[lorder]
bars = ax.bar(range(3), lay_i["n"], color=[lcol[k] for k in lorder],
              edgecolor="black", linewidth=0.5, width=0.62)
for b, v in zip(bars, lay_i["n"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, str(int(v)), ha="center", fontsize=6.8)
ax.set_xticks(range(3)); ax.set_xticklabels([lnice[k] for k in lorder], fontsize=6.5)
ax.set_ylabel("Number of compounds"); ax.set_ylim(0, 500)
ax.set_title("Consensus layer size", fontsize=7.5)
pu.label_panel(ax, "A", x=-0.22)

# (B) composition by DILIrank category
ax = axes[1]
x = np.arange(3); w = 0.36
ax.bar(x - w / 2, lay_i["n_vLess"], w, label="vLess", color=OI["sky_blue"],
       edgecolor="black", linewidth=0.4)
ax.bar(x + w / 2, lay_i["n_amb"], w, label="Ambiguous", color=OI["orange"],
       edgecolor="black", linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels([lnice[k] for k in lorder], fontsize=6.5)
ax.set_ylabel("Number of compounds"); ax.legend(fontsize=6.3, loc="upper left")
ax.set_title("Composition by DILIrank category", fontsize=7.5)
pu.label_panel(ax, "B", x=-0.22)

# (C) vote histogram (n_high over all 659)
ax = axes[2]
vc = sets["n_high"].value_counts().reindex(range(8), fill_value=0)
bcol = [OI["blue"] if k == 0 else OI["vermillion"] if k >= 6 else OI["orange"]
        for k in vc.index]
ax.bar(vc.index, vc.values, color=bcol, edgecolor="black", linewidth=0.4, width=0.72)
for k, v in zip(vc.index, vc.values):
    if v > 0:
        ax.text(k, v + 6, str(int(v)), ha="center", fontsize=6.2)
ax.set_xlabel("High-risk vote count $n_{high}$ (of 7 models)")
ax.set_ylabel("Number of compounds"); ax.set_xticks(range(8))
ax.set_ylim(0, max(vc.values) * 1.18)
ax.set_title("Vote-count distribution, all 659", fontsize=7.5)
ax.annotate("HIGH\n($\\geq$6 votes)", xy=(6, 40), xytext=(4.6, max(vc.values) * 0.62),
            fontsize=6.2, color=OI["vermillion"], ha="center",
            arrowprops=dict(arrowstyle="->", color=OI["vermillion"], lw=0.7))
pu.label_panel(ax, "C", x=-0.20)

fig.suptitle("Figure S3. Consensus layering of the 659 uncategorised compounds "
             "(each model contributes one top-10% vote)", fontsize=8.5)
pu.save_figure(fig, FIG / "FigS3_consensus_layers", formats=("pdf", "png"), dpi=600)
plt.close(fig)
P("[S3] FigS3_consensus_layers")


# ======================================================================
# FigS4 — label-encoding sensitivity
# ======================================================================
lab = pd.read_csv(OUT / "S3_label_sensitivity.csv")
LNICE = {"L1": "L1\nMost vs No\n(n=556)", "L2": "L2\n(Most+Less) vs No\n(n=884)",
         "L3": "L3\n3-class ordinal\n(n=884)", "L4-best": "L4\n4-class + folded\n(n=556)"}
LOR = ["L1", "L2", "L3", "L4-best"]
lab = lab.set_index("scheme").loc[LOR].reset_index()

fig, ax = plt.subplots(figsize=pu.figsize(pu.SINGLE_COL_MM, aspect=1.0),
                       layout="constrained")
x = np.arange(len(lab))
lcol = {"L1": OI["blue"], "L2": OI["sky_blue"], "L3": OI["yellow"], "L4-best": OI["bluish_green"]}
bars = ax.bar(x, lab["auc_mean"], color=[lcol[s] for s in lab["scheme"]],
              edgecolor="black", linewidth=0.5, width=0.6)
ci_lo = lab["ci_lo"].fillna(lab["auc_mean"]); ci_hi = lab["ci_hi"].fillna(lab["auc_mean"])
ax.errorbar(x, lab["auc_mean"], yerr=[lab["auc_mean"] - ci_lo, ci_hi - lab["auc_mean"]],
            fmt="none", ecolor="black", capsize=3.5, lw=1.0)
for b, v, hi in zip(bars, lab["auc_mean"], ci_hi):
    ax.text(b.get_x() + b.get_width() / 2, hi + 0.007, f"{v:.3f}",
            ha="center", va="bottom", fontsize=6.6)
ax.axhline(0.8046, ls="--", color="gray", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels([LNICE[s] for s in lab["scheme"]], fontsize=6.4)
ax.set_ylabel("5-fold CV AUROC (Most vs No)")
ax.set_ylim(0.68, 0.88)
ax.set_title("Figure S4. Label-encoding sensitivity", fontsize=9.0, pad=10)
pu.label_panel(ax, "A", x=-0.16, y=1.10)
ax.text(0.98, 0.97, "L1 and L4 \u2265 L2, L3", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.4,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.5))
pu.save_figure(fig, FIG / "FigS4_label_sensitivity", formats=("pdf", "png"), dpi=600)
plt.close(fig)
P("[S4] FigS4_label_sensitivity")


# ======================================================================
# FigS5 — placement of the 331 Ambiguous compounds
# ======================================================================
pl = pd.read_csv(OUT / "S3_ambiguous_placement.csv")
fig, axes = plt.subplots(1, 2, figsize=pu.figsize(pu.DOUBLE_COL_MM, height_mm=78),
                         layout="constrained", gridspec_kw={"width_ratios": [1.25, 1.0]})

ax = axes[0]
bins = np.linspace(0, 1, 41)
for col, colr, lb in [("P_Most", OI["vermillion"], "$P$(Most)"),
                      ("P_Less", OI["orange"], "$P$(Less)"),
                      ("P_No", OI["blue"], "$P$(No)")]:
    ax.hist(pl[col], bins=bins, histtype="step", lw=1.4, color=colr, label=lb)
for col, colr in [("P_Most", OI["vermillion"]), ("P_Less", OI["orange"]), ("P_No", OI["blue"])]:
    ax.axvline(pl[col].mean(), ls="--", color=colr, lw=0.9)
ax.set_xticks(np.arange(0.0, 1.01, 0.2))
ax.set_xlabel("Out-of-fold predicted probability")
ax.set_ylabel("Number of Ambiguous compounds")
ax.legend(fontsize=6.5, loc="upper center")
ax.set_title("Model placement of the 331 Ambiguous compounds", fontsize=7.8)
pu.label_panel(ax, "A", x=-0.13)

ax = axes[1]
am = pl["argmax"].value_counts().reindex(["Most", "Less", "No", "Amb"], fill_value=0)
acol = {"Most": OI["vermillion"], "Less": OI["orange"], "No": OI["blue"],
        "Amb": OI["yellow"]}
bars = ax.bar(range(len(am)), am.values, color=[acol[k] for k in am.index],
              edgecolor="black", linewidth=0.5, width=0.6)
for b, v in zip(bars, am.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, str(int(v)), ha="center", fontsize=6.5)
ax.set_xticks(range(len(am)))
ax.set_xticklabels(["Most", "Less", "No", "Ambiguous"], fontsize=6.6)
ax.set_ylabel("Number of compounds")
ax.set_title("Argmax assignment", fontsize=7.8)
ax.text(0.03, 0.95,
        f"mean $P$(Less) = {pl['P_Less'].mean():.3f}\nmean $P$(No)   = {pl['P_No'].mean():.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.3,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.5))
pu.label_panel(ax, "B", x=-0.16)

fig.suptitle("Figure S5. The Ambiguous category is not resolved by the model into a single risk direction",
             fontsize=8.5)
pu.save_figure(fig, FIG / "FigS5_ambiguous_placement", formats=("pdf", "png"), dpi=600)
plt.close(fig)
P("[S5] FigS5_ambiguous_placement")


# ======================================================================
# FigS6 — rank stability of the 7-drug panel across 36 combinations (S11)
# ======================================================================
s11 = pd.read_csv(OUT / "S11_rank_summary.csv")
ROLE_ORDER = ["SMX", "CPZ", "CAB", "MBZ", "OXY", "TOL", "MET"]
NAME = {"CPZ": "Chlorpromazine", "SMX": "Sulfamethizole", "CAB": "Cabotegravir",
        "MBZ": "Mebendazole", "OXY": "Oxybutynin", "TOL": "Tolterodine",
        "MET": "Metformin"}
ROLE_COL = {"SMX": OI["vermillion"], "CPZ": OI["vermillion"], "CAB": OI["vermillion"],
            "MBZ": OI["vermillion"], "OXY": OI["bluish_green"],
            "TOL": OI["bluish_green"], "MET": OI["bluish_green"]}
ROLE_MK = {"SMX": "o", "CPZ": "D", "CAB": "s", "MBZ": "^",
           "OXY": "o", "TOL": "D", "MET": "s"}
s11 = s11.set_index("drug").loc[ROLE_ORDER].reset_index()

fig, ax = plt.subplots(figsize=pu.figsize(pu.DOUBLE_COL_MM, height_mm=72),
                       layout="constrained")
y = np.arange(len(s11))[::-1]
for yi, (_, r) in zip(y, s11.iterrows()):
    ax.plot([r["rank_min"], r["rank_max"]], [yi, yi], color=ROLE_COL[r["drug"]],
            lw=2.2, solid_capstyle="round", alpha=0.55)
    ax.plot(r["rank_mean"], yi, marker=ROLE_MK[r["drug"]], color=ROLE_COL[r["drug"]],
            markersize=5.5, markeredgecolor="black", markeredgewidth=0.4, zorder=3)
ax.axvline(66, ls="--", color="gray", lw=0.9)
ax.text(70, 1.5, "Top 10%\n(rank 66)", fontsize=6.2, color="gray", va="center")
ax.set_yticks(y); ax.set_yticklabels([NAME[d] for d in s11["drug"]], fontsize=6.8)
ax.set_xlabel("Rank among the 659 uncategorised compounds (across 36 model combinations)")
ax.set_xlim(0, 660)
ax.set_title("Figure S6. Rank stability of the validation panel\n"
             "(min\u2013max range over 36 = 4 label \u00d7 3 classifier \u00d7 3 descriptor combinations)",
             fontsize=8.8)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], marker="o", color=OI["vermillion"], ls="", label="high-risk candidate"),
    Line2D([], [], marker="o", color=OI["bluish_green"], ls="", label="negative control"),
    Line2D([], [], color="0.5", lw=2.0, alpha=0.6, label="min\u2013max range")],
    fontsize=6.4, loc="lower left", bbox_to_anchor=(1.01, 0.0))
pu.save_figure(fig, FIG / "FigS6_rank_stability", formats=("pdf", "png"), dpi=600)
plt.close(fig)
P("[S6] FigS6_rank_stability")

(OUT / "S13_figures_v2.log").write_text("\n".join(log), encoding="utf-8")
P("\nS13 完成。")
