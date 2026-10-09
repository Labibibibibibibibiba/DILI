# -*- coding: utf-8 -*-
"""
S14 — Final, condensed supplementary figures + modified main Figure 3.

Goal: keep DILI_1 strictly a *supplementary* layer to the "药1" main paper
(Figure 1–3, 4 panels each = 12 panels). Condense the previous 6 supplementary
figures (~13 panels) into 2 multi-panel figures:

  Figure S1  Ranking consistency
     (A) pairwise Spearman heat map of the 7 scoring configurations
     (B) top-10% set overlap (Jaccard)
     (C) consensus layer size + category composition

  Figure S2  Sensitivity and robustness
     (A) descriptor-space 5-fold CV AUROC (F1–F5)
     (B) label-encoding 5-fold CV AUROC (L1–L4)
     (C) rank range of the validation panel across 36 combinations
     (D) placement of the 331 Ambiguous compounds

  Figure3_v2  main Figure 3 with panel D annotated by the 36-combination rank
              range (independent robustness scale) — a CONTROL copy is also
              written so the user can compare before replacing.

Style: strictly aligned with "药1" (D:\\DILI\\paper\\figures\\Figure1-3):
  Okabe-Ito palette, Arial 8pt, no top/right spines, A/B/C/D panel labels,
  English labels, 600 dpi, PDF+PNG.

Outputs -> D:\\DILI_1\\figures_final\\
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc
import plot_utils as pu

pu.apply_paper_style()

OUT = dc.OUT
FIG = dc.DILI1 / "figures_final"
FIG.mkdir(parents=True, exist_ok=True)
OI = pu.OKABE_ITO

log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

P("=" * 72); P("S14 — condensed supplementary figures + modified Figure 3"); P("=" * 72)

SHORT = {
    "L1_F1_A1": "RF/Morgan",
    "L1_F3_A1": "RF/M+2D",
    "L1_F2_A1": "RF/2D",
    "L1_F3_A2": "XGB/M+2D",
    "L1_F3_A3": "LR/M+2D",
    "L1_F4_A1": "RF/3D",
    "L1_F5_A1": "RF/M+2D+3D",
}


# ======================================================================
# Figure S1 — Ranking consistency  (3 panels: A, B, C)
# ======================================================================
sp = pd.read_csv(OUT / "S5_pairwise_spearman.csv", index_col=0)
jac = pd.read_csv(OUT / "S5_jaccard_top10.csv", index_col=0)
lay = pd.read_csv(OUT / "S5_consensus_layers.csv")
lbl = [SHORT[c] for c in sp.columns]

fig, axes = plt.subplots(1, 3, figsize=pu.figsize(pu.DOUBLE_COL_MM, height_mm=68),
                         layout="constrained",
                         gridspec_kw={"width_ratios": [1.15, 1.15, 0.95]})

ax = axes[0]
im = ax.imshow(sp.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
ax.set_xticks(range(len(lbl))); ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=5.8)
ax.set_yticks(range(len(lbl))); ax.set_yticklabels(lbl, fontsize=5.8)
for i in range(len(sp)):
    for j in range(len(sp)):
        v = sp.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.9,
                color="white" if abs(v) > 0.55 else "black")
ax.set_title("Spearman $\\rho$", fontsize=7.2)
cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02)
cb.ax.tick_params(labelsize=5.4)
pu.label_panel(ax, "A", x=-0.32)

ax = axes[1]
im = ax.imshow(jac.values, cmap="Blues", vmin=0, vmax=1, aspect="equal")
ax.set_xticks(range(len(lbl))); ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=5.8)
ax.set_yticks(range(len(lbl))); ax.set_yticklabels(lbl, fontsize=5.8)
for i in range(len(jac)):
    for j in range(len(jac)):
        v = jac.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.9,
                color="white" if v > 0.55 else "black")
ax.set_title("Top-10% Jaccard", fontsize=7.2)
cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02)
cb.ax.tick_params(labelsize=5.4)
pu.label_panel(ax, "B", x=-0.32)

ax = axes[2]
lorder = ["Consensus_HIGH", "Discordant", "Consensus_LOW"]
lnice = {"Consensus_HIGH": "HIGH", "Discordant": "Discordant", "Consensus_LOW": "LOW"}
lcol = {"Consensus_HIGH": OI["vermillion"], "Discordant": OI["orange"], "Consensus_LOW": OI["blue"]}
lay_i = lay.set_index("layer").loc[lorder]
x = np.arange(3); w = 0.36
ax.bar(x - w / 2, lay_i["n_vLess"], w, label="vLess", color=OI["sky_blue"],
       edgecolor="black", linewidth=0.4)
ax.bar(x + w / 2, lay_i["n_amb"], w, label="Ambiguous", color=OI["orange"],
       edgecolor="black", linewidth=0.4)
for xi, n in zip(x, lay_i["n"]):
    ax.text(xi, max(lay_i.loc[lorder[xi], "n_vLess"], lay_i.loc[lorder[xi], "n_amb"]) + 10,
            f"n={int(n)}", ha="center", fontsize=5.8)
ax.set_xticks(x); ax.set_xticklabels([lnice[k] for k in lorder], fontsize=6.2)
ax.set_ylabel("Number of compounds", fontsize=6.6)
ax.set_ylim(0, 520)
ax.legend(fontsize=5.6, loc="upper left", frameon=False)
ax.set_title("Consensus layers", fontsize=7.2)
pu.label_panel(ax, "C", x=-0.24)

fig.suptitle("Figure S1. Consistency of the risk ranking across modelling configurations",
             fontsize=8.6, y=1.04)
pu.save_figure(fig, FIG / "FigS1_consistency", formats=("pdf", "png"), dpi=600)
plt.close(fig)
P("[S1] FigS1_consistency")


# ======================================================================
# Figure S2 — Sensitivity and robustness (4 panels: A, B, C, D)
# ======================================================================
cmp_df = pd.read_csv(OUT / "S4_descriptor_comparison.csv")
lab = pd.read_csv(OUT / "S3_label_sensitivity.csv")
s11 = pd.read_csv(OUT / "S11_rank_summary.csv")
pl = pd.read_csv(OUT / "S3_ambiguous_placement.csv")

fig, axes = plt.subplots(2, 2, figsize=pu.figsize(pu.DOUBLE_COL_MM, height_mm=132),
                         layout="constrained",
                         gridspec_kw={"hspace": 0.30, "wspace": 0.24})

# (A) descriptor space
NICE = {"F1_Morgan": "F1\nMorgan", "F2_2Ddesc": "F2\n2D desc", "F3_Morgan+2D": "F3\nMorgan+2D",
        "F4_3Donly": "F4\n3D only", "F5_2D+3D": "F5\n2D+3D"}
order = ["F1_Morgan", "F2_2Ddesc", "F3_Morgan+2D", "F4_3Donly", "F5_2D+3D"]
cmp_df = cmp_df.set_index("feature_set").loc[order].reset_index()
ax = axes[0, 0]
x = np.arange(len(cmp_df))
cols = [OI["blue"] if "3D" not in s else OI["orange"] for s in cmp_df["feature_set"]]
bars = ax.bar(x, cmp_df["auc_mean"], color=cols, edgecolor="black", linewidth=0.5, width=0.62)
for b, s in zip(bars, cmp_df["feature_set"]):
    if "3D" in s: b.set_hatch("//")
ax.errorbar(x, cmp_df["auc_mean"],
            yerr=[cmp_df["auc_mean"] - cmp_df["ci_lo"], cmp_df["ci_hi"] - cmp_df["auc_mean"]],
            fmt="none", ecolor="black", capsize=2.5, lw=0.9)
ax.axhline(0.8046, ls="--", color="gray", lw=0.8)
for b, v in zip(bars, cmp_df["auc_mean"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=5.6)
ax.set_xticks(x); ax.set_xticklabels([NICE[s] for s in cmp_df["feature_set"]], fontsize=5.8)
ax.set_ylabel("5-fold CV AUROC", fontsize=6.6); ax.set_ylim(0.55, 0.90)
ax.legend(handles=[Patch(facecolor=OI["blue"], edgecolor="black", label="2D"),
                   Patch(facecolor=OI["orange"], edgecolor="black", hatch="//", label="+3D")],
          fontsize=5.4, loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.99))
ax.text(0.98, 0.88, "$\\Delta$AUROC(F5$-$F3)$=-0.015$", transform=ax.transAxes,
        ha="right", va="top", fontsize=5.4)
ax.set_title("Descriptor space", fontsize=7.2)
pu.label_panel(ax, "A", x=-0.22)

# (B) label encoding
LNICE = {"L1": "L1", "L2": "L2", "L3": "L3", "L4-best": "L4"}
LOR = ["L1", "L2", "L3", "L4-best"]
lab = lab.set_index("scheme").loc[LOR].reset_index()
ax = axes[0, 1]
x = np.arange(len(lab))
lcol = {"L1": OI["blue"], "L2": OI["sky_blue"], "L3": OI["yellow"], "L4-best": OI["bluish_green"]}
bars = ax.bar(x, lab["auc_mean"], color=[lcol[s] for s in lab["scheme"]],
              edgecolor="black", linewidth=0.5, width=0.6)
ci_lo = lab["ci_lo"].fillna(lab["auc_mean"]); ci_hi = lab["ci_hi"].fillna(lab["auc_mean"])
ax.errorbar(x, lab["auc_mean"], yerr=[lab["auc_mean"] - ci_lo, ci_hi - lab["auc_mean"]],
            fmt="none", ecolor="black", capsize=2.5, lw=0.9)
for b, v in zip(bars, lab["auc_mean"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=5.6)
ax.axhline(0.8046, ls="--", color="gray", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([LNICE[s] for s in lab["scheme"]], fontsize=6.2)
ax.set_ylabel("5-fold CV AUROC", fontsize=6.6); ax.set_ylim(0.68, 0.88)
ax.set_title("Label-encoding scheme", fontsize=7.2)
pu.label_panel(ax, "B", x=-0.22)

# (C) rank range across 36 combos
ROLE_ORDER = ["SMX", "CPZ", "CAB", "MBZ", "OXY", "TOL", "MET"]
NAME = {"CPZ": "Chlorpromazine", "SMX": "Sulfamethizole", "CAB": "Cabotegravir",
        "MBZ": "Mebendazole", "OXY": "Oxybutynin", "TOL": "Tolterodine", "MET": "Metformin"}
ROLE_COL = {"SMX": OI["vermillion"], "CPZ": OI["vermillion"], "CAB": OI["vermillion"],
            "MBZ": OI["vermillion"], "OXY": OI["bluish_green"],
            "TOL": OI["bluish_green"], "MET": OI["bluish_green"]}
ROLE_MK = {"SMX": "o", "CPZ": "D", "CAB": "s", "MBZ": "^", "OXY": "o", "TOL": "D", "MET": "s"}
s11 = s11.set_index("drug").loc[ROLE_ORDER].reset_index()
ax = axes[1, 0]
y = np.arange(len(s11))[::-1]
for yi, (_, r) in zip(y, s11.iterrows()):
    ax.plot([r["rank_min"], r["rank_max"]], [yi, yi], color=ROLE_COL[r["drug"]],
            lw=2.0, solid_capstyle="round", alpha=0.55)
    ax.plot(r["rank_mean"], yi, marker=ROLE_MK[r["drug"]], color=ROLE_COL[r["drug"]],
            markersize=4.6, markeredgecolor="black", markeredgewidth=0.4, zorder=3)
ax.axvline(66, ls="--", color="gray", lw=0.8)
ax.text(72, len(s11) - 0.5, "Top 10%", fontsize=5.4, color="gray", va="top")
ax.set_yticks(y); ax.set_yticklabels([NAME[d] for d in s11["drug"]], fontsize=5.8)
ax.set_xlabel("Rank among 659 uncategorised compounds", fontsize=6.4)
ax.set_xlim(0, 660)
ax.legend(handles=[
    Line2D([], [], marker="o", color=OI["vermillion"], ls="", label="high-risk candidate"),
    Line2D([], [], marker="o", color=OI["bluish_green"], ls="", label="negative control")],
    fontsize=5.2, loc="upper right", frameon=False)
ax.set_title("Rank range (36 combinations)", fontsize=7.2)
pu.label_panel(ax, "C", x=-0.28)

# (D) Ambiguous placement
ax = axes[1, 1]
bins = np.linspace(0, 1, 41)
for col, colr, lb in [("P_Most", OI["vermillion"], "$P$(Most)"),
                      ("P_Less", OI["orange"], "$P$(Less)"),
                      ("P_No", OI["blue"], "$P$(No)")]:
    ax.hist(pl[col], bins=bins, histtype="step", lw=1.2, color=colr, label=lb)
ax.set_xlabel("Out-of-fold predicted probability", fontsize=6.4)
ax.set_ylabel("Number of compounds", fontsize=6.6)
ax.legend(fontsize=5.4, loc="upper center", frameon=False, ncol=3, columnspacing=0.8)
ax.set_title("Placement of the 331 Ambiguous compounds", fontsize=7.2)
pu.label_panel(ax, "D", x=-0.22)

fig.suptitle("Figure S2. Sensitivity of the risk ranking to modelling choices",
             fontsize=8.6, y=1.02)
pu.save_figure(fig, FIG / "FigS2_sensitivity", formats=("pdf", "png"), dpi=600)
plt.close(fig)
P("[S2] FigS2_sensitivity")


# ======================================================================
# Figure 3 (modified) — panel D annotated with 36-combination rank range
# ======================================================================
US = json.load(open(r"D:/DILI/data/unknown_summary_v2.json"))
rank_df = pd.read_excel(r"D:/DILI/data/DILI_unknown_prediction_rank_v2.xlsx")
tail_df = pd.read_excel(r"D:/DILI/data/DILI_tail_enrichment_analysis_v2.xlsx")

vless = rank_df[rank_df["DILI_category"] == "vLess-DILI-concern"]["DILI_risk_score"].values
ambig = rank_df[rank_df["DILI_category"] == "Ambiguous-DILI-concern"]["DILI_risk_score"].values
all_scores = rank_df["DILI_risk_score"].values
P_MW = US["mannwhitney_p"]
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
top5_threshold = float(np.sort(all_scores)[::-1][31])

# rank range from S11 (fixed-split 444 scale) — independent of the v2 scores
RANK_RANGE = s11.set_index("drug")[["rank_min", "rank_max"]].to_dict("index")
_rk = {"Chlorpromazine": "CPZ", "Sulfamethizole": "SMX", "Cabotegravir": "CAB",
       "Mebendazole": "MBZ", "Oxybutynin": "OXY", "Tolterodine": "TOL", "Metformin": "MET"}

fig = plt.figure(figsize=pu.figsize(pu.DOUBLE_COL_MM, height_mm=118))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0],
                      hspace=0.55, wspace=0.45, top=0.90, bottom=0.12)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

# A
cat_colors = [pu.CATEGORY_COLORS["vLess-DILI-concern"], pu.CATEGORY_COLORS["Ambiguous-DILI-concern"]]
parts = axA.violinplot([vless, ambig], positions=[0, 1], showmeans=False,
                       showmedians=False, showextrema=False)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(cat_colors[i]); pc.set_alpha(0.45); pc.set_edgecolor("none")
axA.boxplot([vless, ambig], positions=[0, 1], widths=0.12, patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.0),
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8))
axA.set_xticks([0, 1]); axA.set_xticklabels(["vLess\n(n=328)", "Ambiguous\n(n=331)"], fontsize=6.5)
axA.set_ylabel("DILI risk score"); axA.set_ylim(0, 1)
axA.tick_params(axis="x", labelsize=6.5); axA.tick_params(axis="y", labelsize=6)
pu.add_significance(axA, 0, 1, 0.95, "P = {:.3f}\n(Mann-Whitney U)".format(P_MW))
pu.label_panel(axA, "A")

# B
bins = np.arange(0.0, 0.9, 0.04)
axB.hist([vless, ambig], bins=bins, stacked=True, color=cat_colors, alpha=0.75,
         edgecolor="white", linewidth=0.3, label=["vLess", "Ambiguous"])
axB.axvline(top5_threshold, color=OI["vermillion"], linewidth=1.2, linestyle="--")
axB.text(top5_threshold + 0.01, axB.get_ylim()[1] * 0.9, "Top 5%", fontsize=6,
         color=OI["vermillion"], ha="left", fontweight="bold")
axB.set_xlabel("DILI risk score"); axB.set_ylabel("Number of compounds")
axB.tick_params(axis="x", labelsize=6); axB.tick_params(axis="y", labelsize=6)
axB.legend(fontsize=6, frameon=False, loc="upper left")
pu.label_panel(axB, "B")

# C
labels_C = tail_df["Top_fraction"].tolist()
ors_C = tail_df["Odds_ratio"].values; ps_C = tail_df["Fisher_p"].values
ypos_C = np.arange(len(labels_C))[::-1]
colors_C = [OI["vermillion"] if i == 0 else "#888780" for i in range(len(labels_C))]
axC.scatter(ors_C, ypos_C, s=60, color=colors_C, zorder=3)
lo_C = tail_df["OR_CI_low"].values; hi_C = tail_df["OR_CI_high"].values
axC.errorbar(ors_C, ypos_C, xerr=[np.clip(ors_C - lo_C, 0, None), np.clip(hi_C - ors_C, 0, None)],
             fmt="none", ecolor="#888780", elinewidth=0.8, capsize=2, zorder=2)
axC.axvline(1, color="#888780", linewidth=0.8, linestyle="--")
axC.set_yticks(ypos_C); axC.set_yticklabels(labels_C, fontsize=6.5)
axC.set_xlabel("Odds ratio (vLess enrichment)"); axC.set_xscale("log"); axC.set_xlim(0.7, 7.5)
axC.tick_params(axis="x", labelsize=6)
for yi, or_, p in zip(ypos_C, ors_C, ps_C):
    axC.text(or_ * 1.10, yi + 0.20, f"OR={or_:.2f}, P={p:.3f}", fontsize=5.2, ha="left", va="bottom",
             color=OI["vermillion"] if or_ > 3 else "#444441",
             fontweight="bold" if or_ > 3 else "normal")
axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)
pu.label_panel(axC, "C")

# D — with rank-range annotation
drugs_sorted = sorted(validation_drugs, key=lambda x: -x[1])
for i, (name, score, group) in enumerate(drugs_sorted):
    color = pu.DRUG_GROUP_COLORS[group]; marker = pu.DRUG_GROUP_MARKERS[group]
    axD.scatter([score], [i], s=60, color=color, marker=marker,
                edgecolor="black", linewidth=0.5, zorder=3)
    # drug name placed just right of the marker (shorter offset to leave room)
    axD.text(score + 0.025, i, name, fontsize=6, va="center", ha="left")
    # rank-range annotation in a separate right-hand column (independent scale)
    rr = RANK_RANGE.get(_rk[name])
    if rr is not None:
        axD.text(0.60, i, f"{int(rr['rank_min'])}\u2013{int(rr['rank_max'])}",
                 transform=axD.get_yaxis_transform(), fontsize=5.0, va="center", ha="left",
                 color="#5F5E5A")
axD.axvline(top5_threshold, color=OI["vermillion"], linewidth=1.0, linestyle="--")
axD.text(top5_threshold + 0.01, len(drugs_sorted) - 0.5, "Top 5%", fontsize=5.5,
         color=OI["vermillion"], ha="left")
axD.text(0.60, len(drugs_sorted) - 0.5, "rank range\n(all 36 combos)", fontsize=5.0,
         color="#5F5E5A", va="bottom", ha="left", transform=axD.get_yaxis_transform())
axD.set_yticks([]); axD.set_ylim(-0.8, len(drugs_sorted) - 0.2)
axD.set_xlabel("DILI risk score", fontsize=6); axD.set_xlim(0.2, 1.32)
axD.tick_params(axis="x", labelsize=6)
axD.spines["top"].set_visible(False); axD.spines["right"].set_visible(False)
axD.spines["left"].set_visible(False)
legend_D = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=pu.DRUG_GROUP_COLORS["positive_control"],
           markersize=6, label="Positive control"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=pu.DRUG_GROUP_COLORS["high_risk_candidate"],
           markersize=6, label="High-risk candidate"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=pu.DRUG_GROUP_COLORS["negative_control"],
           markersize=6, label="Negative control"),
]
axD.legend(handles=legend_D, fontsize=5.5, frameon=False, loc="lower left",
           bbox_to_anchor=(0.0, 0.0))
pu.label_panel(axD, "D")

posD = axD.get_position()
fig.text(0.5 * (posD.x0 + posD.x1), posD.y0 - 0.100,
         "Diclofenac (out-of-pool positive control, Most-DILI-concern) is not scored by the model.\n"
         "Right-hand numbers: rank range across all 36 modelling configurations (independent robustness scale).",
         fontsize=5.3, ha="center", va="top", color="#5F5E5A", style="italic", linespacing=1.4)

pu.save_figure(fig, FIG / "Figure3_v2", formats=("pdf", "png"), dpi=600, tight=False)
plt.close(fig)
P("[Fig3] Figure3_v2 (with rank-range annotation in panel D)")

# save a clean copy of the original Figure3 for side-by-side comparison
import shutil
for ext in ("png", "pdf"):
    src = dc.DILI / "paper" / "figures" / f"Figure3.{ext}"
    if src.exists():
        shutil.copy(src, FIG / f"Figure3_original.{ext}")
P("[Fig3] Figure3_original (unmodified reference copy)")

(OUT / "S14_figures_final.log").write_text("\n".join(log), encoding="utf-8")
P("\nS14 完成。")
