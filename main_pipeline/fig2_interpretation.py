#!/usr/bin/env python3
"""Figure 2 — CORRECTED COPY of D:/DILI/code/fig2_interpretation.py.

Changes vs the original (2026-10-09):
  1. Panel C colouring: each marker now uses its own row's Significant flag
     (original looked it up by fingerprint and mis-coloured FP_698 Cc grey).
  2. Output redirected to D:/DILI_1/figures_regen (D:/DILI is read-only).
"""
import sys
sys.path.insert(0, r"D:/abraca")
import plot_utils as pu
pu.apply_paper_style()

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

DATA = r"D:/DILI/data"

# =============================================================================
# 数据加载（v2：SHAP 用缓存的 556 全数据集结果，保证与富集分析同源）
# =============================================================================
vals = np.load(f"{DATA}/shap_values_556_v2.npy")          # (556, 1031)
feat_names = json.load(open(f"{DATA}/shap_feature_names_v2.json"))
X = pd.read_excel(f"{DATA}/X_features_final.xlsx")
imp_df = pd.read_excel(f"{DATA}/shap_importance_v2.xlsx")
fam = pd.read_excel(f"{DATA}/enrichment_fullfamily_bitidentity.xlsx")

print("SHAP values:", vals.shape, "| families:", fam.shape)

imp = np.abs(vals).mean(axis=0)

# 二值特征判定
def is_binary(col):
    return len(X[col].dropna().unique()) <= 2

# =============================================================================
# 画布 2×2
# =============================================================================
fig = plt.figure(figsize=pu.figsize(pu.DOUBLE_COL_MM, height_mm=130))
gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0],
                      width_ratios=[1.35, 1.0], hspace=0.62, wspace=0.45,
                      top=0.90, bottom=0.09)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

# =============================================================================
# A：SHAP beeswarm
# =============================================================================
top_idx = np.argsort(imp)[::-1][:20]
vals_top = vals[:, top_idx]
names_top = [feat_names[i] for i in top_idx]
# beeswarm 用的特征取值：从 X 中按名字取
data_top = X[names_top].values

cont_names = [n for n in names_top if not is_binary(n)]
cont_vals = X[cont_names].values.ravel() if cont_names else np.array([0.0, 1.0])
vmin, vmax = np.nanpercentile(cont_vals, 5), np.nanpercentile(cont_vals, 95)
norm_cont = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap_cont = plt.colormaps["viridis"]
GRAY, ORANGE_RED = "#D3D1C7", "#D55E00"

for i in range(len(names_top)):
    ypos = len(names_top) - 1 - i
    sh = vals_top[:, i]
    fe = data_top[:, i]
    rng = np.random.default_rng(i)
    jitter = rng.normal(0, 0.18, len(sh))
    if is_binary(names_top[i]):
        colors = [ORANGE_RED if v >= 0.5 else GRAY for v in fe]
    else:
        colors = [cmap_cont(norm_cont(v)) for v in fe]
    axA.scatter(sh, ypos + jitter, s=4, c=colors, alpha=0.6, linewidths=0)

axA.axvline(0, color="#888780", linewidth=0.7)
axA.set_yticks(range(len(names_top)))
axA.set_yticklabels([names_top[-1 - i] for i in range(len(names_top))], fontsize=4.8)
axA.set_xlabel("SHAP value (impact on DILI probability)", fontsize=6.5)
axA.tick_params(axis="x", labelsize=5.5)
sm = plt.cm.ScalarMappable(norm=norm_cont, cmap=cmap_cont)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axA, fraction=0.025, pad=0.015)
cbar.ax.tick_params(labelsize=5)
cbar.set_label("Continuous\nvalue", fontsize=5)
axA.legend(handles=[
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE_RED, markersize=5, label="FP present (1)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GRAY, markersize=5, label="FP absent (0)"),
], loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=5.5, frameon=False)
pu.label_panel(axA, "A", y=1.26)

# =============================================================================
# B：Feature importance（mean |SHAP| Top 20）
# =============================================================================
bar_vals = imp[top_idx][::-1]
bar_names = [names_top[-1 - i] for i in range(len(names_top))]
axB.barh(range(len(bar_names)), bar_vals, color=pu.OKABE_ITO["blue"], height=0.6, edgecolor="none")
axB.set_yticks(range(len(bar_names)))
axB.set_yticklabels(bar_names, fontsize=4.8)
axB.set_xlabel("Mean |SHAP value|", fontsize=6.5)
axB.tick_params(axis="x", labelsize=5.5)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
pu.label_panel(axB, "B")

# =============================================================================
# C：子结构级富集森林图（OR + 95% CI）
#    显示 SHAP 重要指纹位派生、且携带数 >= 10 的子结构；
#    FDR 取自全指纹空间检验族（n = 809），非限定族。
# =============================================================================
important_bits = [f for f in imp_df["Feature"] if str(f).startswith("FP_")]
sel = fam[fam["Fingerprint"].isin(important_bits) & (fam["Count"] >= 10)].copy()
sel = sel.sort_values("OddsRatio", ascending=False)
sel["label"] = sel["Fingerprint"] + "  " + sel["Substructure_SMILES"]
if len(sel) > 13:
    sel = pd.concat([sel.head(9), sel.tail(4)])
sel = sel.iloc[::-1]                       # 让高 OR 在上
print("Panel C 子结构数:", len(sel))
print(sel[["label", "Count", "OddsRatio", "OR_CI_low", "OR_CI_high", "FDR", "Significant"]].to_string(index=False))

ypos = np.arange(len(sel))
# Colour each marker by its OWN significance flag. An earlier version looked the
# flag up by fingerprint; after the iloc[::-1] reversal that returned the first
# *remaining* row for duplicated bits, so FP_698 Cc (OR 3.68, FDR 0.0029, significant)
# was drawn in the non-significant grey. Fixed 2026-10-09.
cols = []
for _, row in sel.iterrows():
    if row["Fingerprint"] == "FP_367":
        cols.append(pu.OKABE_ITO["vermillion"])
    elif bool(row["Significant"]):
        cols.append(pu.OKABE_ITO["blue"])
    else:
        cols.append("#B4B2A9")
ors = sel["OddsRatio"].values
lo = sel["OR_CI_low"].values
hi = sel["OR_CI_high"].values
XMAX = 60.0
axC.errorbar(ors, ypos, xerr=[np.clip(ors - lo, 0, None), np.clip(hi - ors, 0, None)],
             fmt="o", markersize=3.5, color="none", ecolor="#888780",
             elinewidth=0.7, capsize=1.5, zorder=2)
axC.scatter(ors, ypos, s=26, c=cols, zorder=3, edgecolor="none")
axC.axvline(1, color="#888780", linewidth=0.8, linestyle="--")
axC.set_yticks(ypos)
axC.set_yticklabels(sel["label"], fontsize=5.2)
axC.set_xlabel("Odds ratio (DILI-positive enrichment)", fontsize=6.5)
axC.set_xscale("log")
axC.set_xlim(0.05, XMAX)
axC.tick_params(axis="x", labelsize=5.5)
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)
# FP_367 标注
if "FP_367" in list(sel["Fingerprint"]):
    k = list(sel["Fingerprint"]).index("FP_367")
    axC.annotate("cc(c)C\nOR={:.2f}, FDR={:.3f}".format(ors[k], sel["FDR"].iloc[k]),
                 xy=(ors[k], ypos[k]), xytext=(ors[k] * 1.35, ypos[k] + 0.9),
                 fontsize=5, color=pu.OKABE_ITO["vermillion"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=pu.OKABE_ITO["vermillion"], lw=0.8))
axC.text(0.98, 0.02, "FDR: BH within 809-fragment family\nCI extending beyond axis clipped",
         transform=axC.transAxes, fontsize=4.4, ha="right", va="bottom", color="#5F5E5A")
pu.label_panel(axC, "C", y=1.25)

# =============================================================================
# D：Acitretin 结构 + FP_367 片段高亮（RDKit）
# =============================================================================
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image

acitretin = Chem.MolFromSmiles("CC1=CC(=C(C(=C1C=CC(=CC=CC(=CC(=O)O)C)C)C)C)OC")
pat = Chem.MolFromSmarts("cc(c)C")
match = acitretin.GetSubstructMatch(pat)
highlight = list(match) if match else [0, 1, 2, 6]

drawer = rdMolDraw2D.MolDraw2DCairo(500, 380)
drawer.drawOptions().highlightRadius = 0.35
rdMolDraw2D.PrepareAndDrawMolecule(
    drawer, acitretin, highlightAtoms=highlight,
    highlightAtomColors={i: (0.84, 0.35, 0.18) for i in highlight})
drawer.FinishDrawing()
axD.imshow(Image.open(io.BytesIO(drawer.GetDrawingText())))
axD.axis("off")
axD.set_title("Acitretin: FP_367 (cc(c)C)", fontsize=6.5)
pu.label_panel(axD, "D", y=1.25)

# =============================================================================
pu.save_figure(fig, r"D:/DILI_1/figures_regen/Figure2", formats=("pdf", "png"),
               dpi=600, tight=False)
plt.close(fig)
print("Figure2 完成")
