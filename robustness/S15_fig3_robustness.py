# -*- coding: utf-8 -*-
"""
S15 — Main-text Figure 2: robustness of the risk ranking to modelling choices.

Standalone main figure (NOT supplementary) summarising the DILI_1 robustness
audit in four one-to-one panels:

  Panel A  label-encoding scheme      L1–L4   (source: out/S3_label_sensitivity.csv)
  Panel B  descriptor space + 3D      F1–F5   (source: out/S4_descriptor_comparison.csv)
  Panel C  consensus layers + votes   C1 layer size / C2 vote histogram
                                              (source: out/S5_consensus_layers.csv,
                                                       out/S5_high_risk_sets.csv)
  Panel D  36-combination rank range  7 drugs (source: out/S11_rank_summary.csv)

Figure numbering convention (user decision 2026-10-05):
  Figure 1  model derivation
  Figure 2  robustness          <-- this file
  Figure 3  model interpretation
  Figure 4  model validation / application

Style: strictly aligned with "药1" (D:\\DILI\\paper\\figures\\Figure1-3):
  Okabe-Ito palette, Arial 8pt, no top/right spines, A/B/C/D panel labels,
  English labels, 600 dpi, PDF+PNG, 183 mm double column.

Outputs -> D:\\DILI_1\\figures_main\\Figure3_robustness.{pdf,png}
           D:\\DILI_1\\out\\S15_caption.txt
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc
import plot_utils as pu

pu.apply_paper_style()

OUT = dc.OUT
FIG = dc.DILI1 / "figures_main"
FIG.mkdir(parents=True, exist_ok=True)
OI = pu.OKABE_ITO
GREY = "#888888"

log = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)

P("=" * 72)
P("S15 — main Figure 2: robustness of the risk ranking")
P("=" * 72)

# ======================================================================
# Data loading (all read-only, name-joined where relevant)
# ======================================================================
# --- Panel A: label encoding ---
lab = pd.read_csv(OUT / "S3_label_sensitivity.csv")
lab = lab.set_index("scheme")

# --- Panel B: descriptor space ---
desc = pd.read_csv(OUT / "S4_descriptor_comparison.csv")
desc = desc.set_index("feature_set")

# --- Panel C1: consensus layers ---
lay = pd.read_csv(OUT / "S5_consensus_layers.csv")
lay = lay.set_index("layer")

# --- Panel C2: vote histogram ---
votes_df = pd.read_csv(OUT / "S5_high_risk_sets.csv")
vote_counts = votes_df["n_high"].value_counts().sort_index()
vote_hist = np.array([int(vote_counts.get(k, 0)) for k in range(8)])  # bins 0..7

# --- Panel D: 36-combination rank ranges ---
rk = pd.read_csv(OUT / "S11_rank_summary.csv")
rk = rk[(rk["present"] == True) & (~rk["role"].astype(str).str.contains("审计"))].copy()

# Hardcoded English labels (role column is Chinese — never use it as an axis label)
DRUG_EN = {
    "CPZ": "Chlorpromazine",
    "SMX": "Sulfamethizole",
    "CAB": "Cabotegravir",
    "MBZ": "Mebendazole",
    "OXY": "Oxybutynin",
    "TOL": "Tolterodine",
    "MET": "Metformin",
}
# semantic group: positive / high-risk  (warm)  vs  negative control (cool)
DRUG_GROUP = {
    "CPZ": "positive", "SMX": "highrisk", "CAB": "highrisk", "MBZ": "highrisk",
    "OXY": "negative", "TOL": "negative", "MET": "negative",
}
GROUP_COLOR = {
    "positive": OI["vermillion"],
    "highrisk": OI["orange"],
    "negative": OI["blue"],
}
# order: positives/high-risk on top, negatives below (by rank_mean within group)
order = ["CPZ", "SMX", "MBZ", "CAB", "OXY", "TOL", "MET"]
rk = rk.set_index("drug").loc[order].reset_index()


# ======================================================================
# Figure layout: 2 x 2
# ======================================================================
fig = plt.figure(figsize=pu.figsize(pu.DOUBLE_COL_MM, height_mm=150))
gs = fig.add_gridspec(2, 2, hspace=0.72, wspace=0.30,
                      left=0.07, right=0.985, top=0.965, bottom=0.135)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC_outer = gs[1, 0]
axD = fig.add_subplot(gs[1, 1])


# ======================================================================
# Panel A — label-encoding scheme (L1–L4)
# ======================================================================
schemes = ["L1", "L2", "L3", "L4-best"]   # raw scheme ids in the CSV
sdisp = ["L1", "L2", "L3", "L4"]          # display labels
scolors = [OI["blue"], OI["sky_blue"], OI["yellow"], OI["bluish_green"]]
x = np.arange(len(schemes))
auc = np.array([lab.loc[s, "auc_mean"] for s in schemes], dtype=float)
lo = np.array([lab.loc[s, "ci_lo"] for s in schemes], dtype=float)
hi = np.array([lab.loc[s, "ci_hi"] for s in schemes], dtype=float)
# L4 has NaN CI -> no error bar; plot others
has_ci = np.isfinite(lo) & np.isfinite(hi)
err = np.zeros((2, len(schemes)))
err[0, has_ci] = auc[has_ci] - lo[has_ci]
err[1, has_ci] = hi[has_ci] - auc[has_ci]

bars = axA.bar(x, auc, width=0.62, color=scolors, edgecolor="black", linewidth=0.5,
               yerr=err, error_kw=dict(ecolor="black", lw=0.7, capsize=2.2))
for xi, s, a in zip(x, sdisp, auc):
    axA.text(xi, a + 0.006, f"{a:.3f}", ha="center", va="bottom", fontsize=6.6)
    if s == "L4":
        axA.text(xi, a - 0.013, "no CI", ha="center", va="top", fontsize=5.4, color=GREY)

axA.axhline(lab.loc["L1", "auc_mean"], ls="--", lw=0.7, color=GREY, zorder=0)
axA.text(-0.40, lab.loc["L1", "auc_mean"] - 0.004, "L1 baseline", fontsize=5.6,
         color=GREY, ha="left", va="top")
axA.set_xticks(x)
axA.set_xticklabels(["L1\n(Most vs No)", "L2\n(Most+Less)", "L3\n(ordinal)", "L4\n(4-class folded)"],
                    fontsize=6.4)
axA.set_ylabel("Five-fold CV AUROC")
axA.set_ylim(0.68, 0.88)
axA.set_yticks([0.70, 0.75, 0.80, 0.85])
pu.label_panel(axA, "A")


# ======================================================================
# Panel B — descriptor space and 3D (F1–F5)
# ======================================================================
fsets = ["F1_Morgan", "F2_2Ddesc", "F3_Morgan+2D", "F4_3Donly", "F5_2D+3D"]
flabels = ["F1\nMorgan", "F2\n2D desc.", "F3\nM+2D", "F4\n3D only", "F5\n2D+3D"]
x = np.arange(len(fsets))
fauc = np.array([desc.loc[f, "auc_mean"] for f in fsets], dtype=float)
flo = np.array([desc.loc[f, "ci_lo"] for f in fsets], dtype=float)
fhi = np.array([desc.loc[f, "ci_hi"] for f in fsets], dtype=float)
ferr = np.vstack([fauc - flo, fhi - fauc])
fcolors = [OI["blue"], OI["sky_blue"], OI["bluish_green"],
           OI["vermillion"], OI["vermillion"]]
fhatch = [None, None, None, "//", "//"]

for xi, (f, a, c, h) in enumerate(zip(fsets, fauc, fcolors, fhatch)):
    axB.bar(xi, a, width=0.62, color=c, edgecolor="black", linewidth=0.5,
            hatch=h, yerr=ferr[:, xi:xi + 1], error_kw=dict(ecolor="black", lw=0.7, capsize=2.2))
    axB.text(xi, a + 0.008, f"{a:.3f}", ha="center", va="bottom", fontsize=6.4)

axB.set_xticks(x)
axB.set_xticklabels(flabels, fontsize=6.4)
axB.set_ylabel("Five-fold CV AUROC")
axB.set_ylim(0.55, 0.90)
axB.set_yticks([0.60, 0.70, 0.80, 0.90])
axB.text(0.5, -0.30, r"$\Delta$AUROC (F5$-$F3) $=$ $-0.015$  [$-0.024,\ -0.005$]",
         transform=axB.transAxes, fontsize=6.1, ha="center", va="top",
         bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=GREY, lw=0.6))
axB.legend(handles=[Patch(facecolor=OI["vermillion"], edgecolor="black",
                          hatch="//", label="3D-containing")],
           loc="upper right", fontsize=5.8, frameon=False, handlelength=1.4)
pu.label_panel(axB, "B")


# ======================================================================
# Panel C — consensus layers (C1) + vote histogram (C2)
# ======================================================================
sub = axC_outer.subgridspec(1, 2, wspace=0.62)
axC1 = fig.add_subplot(sub[0, 0])
axC2 = fig.add_subplot(sub[0, 1])

# --- C1: layer sizes ---
layers = ["Consensus_HIGH", "Discordant", "Consensus_LOW"]
lvals = [int(lay.loc[l, "n"]) for l in layers]
llabels = ["HIGH\n($\\geq$6 votes)", "Discordant", "LOW\n(0 votes)"]
lcolors = [OI["vermillion"], OI["orange"], OI["blue"]]
xc = np.arange(3)
axC1.bar(xc, lvals, width=0.62, color=lcolors, edgecolor="black", linewidth=0.5)
for xi, v in zip(xc, lvals):
    axC1.text(xi, v + 8, str(v), ha="center", va="bottom", fontsize=6.6)
axC1.set_xticks(xc)
axC1.set_xticklabels(llabels, fontsize=5.8)
axC1.set_ylabel("Uncategorised compounds", fontsize=7)
axC1.set_ylim(0, 510)
axC1.set_yticks([0, 100, 200, 300, 400, 500])
axC1.tick_params(labelsize=6.2)

# --- C2: vote histogram (n_high) ---
xv = np.arange(8)
vcolors = [OI["blue"] if k == 0 else (OI["vermillion"] if k >= 6 else OI["orange"])
           for k in range(8)]
axC2.bar(xv, vote_hist, width=0.68, color=vcolors, edgecolor="black", linewidth=0.5)
for xi, v in zip(xv, vote_hist):
    axC2.text(xi, v + 10, str(int(v)), ha="center", va="bottom", fontsize=5.8)
axC2.annotate("no compound\nin all 7",
              xy=(7, 6), xytext=(5.15, 190), fontsize=5.6, color=GREY, ha="center",
              arrowprops=dict(arrowstyle="->", color=GREY, lw=0.6))
axC2.set_xticks(xv)
axC2.set_xticklabels([str(k) for k in range(8)], fontsize=6.2)
axC2.set_xlabel("Votes $n_{high}$ (of 7 models)", fontsize=6.6)
axC2.set_ylabel("Compounds", fontsize=7)
axC2.set_ylim(0, 510)
axC2.set_yticks([0, 100, 200, 300, 400, 500])
axC2.tick_params(labelsize=6.2)

pu.label_panel(axC1, "C")


# ======================================================================
# Panel D — 36-combination rank range for the 7 validation drugs
# ======================================================================
n = len(rk)
yv = np.arange(n)[::-1]   # first row on top
for i, row in enumerate(rk.itertuples()):
    yi = yv[i]
    c = GROUP_COLOR[DRUG_GROUP[row.drug]]
    axD.hlines(yi, row.rank_min, row.rank_max, color=c, lw=2.4, alpha=0.85,
               zorder=2, clip_on=False)
    axD.plot([row.rank_min, row.rank_max], [yi, yi], "|", color=c, ms=7, mew=1.1, zorder=3)
    axD.plot([row.rank_mean], [yi], "o", color="white", mec=c, mew=1.2, ms=4.4, zorder=4)

# Top-10% threshold (≈10% of 659)
axD.axvline(66, ls=":", lw=0.8, color=GREY, zorder=1)
axD.text(74, -0.55, "Top 10%", fontsize=5.6, color=GREY, ha="left", va="bottom")

axD.set_yticks(yv)
axD.set_yticklabels([DRUG_EN[d] for d in rk["drug"]], fontsize=6.4)
# extend xlim well past 659 so the right-hand annotation column has clear room
axD.set_xlim(0, 980)
axD.set_ylim(-0.7, n - 0.3)
axD.set_xlabel("Rank among the 659 uncategorised compounds", fontsize=6.8)
axD.set_xticks([0, 100, 200, 300, 400, 500, 600])
axD.tick_params(axis="x", labelsize=6.2)

# right-hand annotation column placed in the reserved x-range [700, 980]
axD.text(700, n - 0.35, "min\u2013max (36 = 4\u00d73\u00d73)", fontsize=5.6,
         color=GREY, ha="left", va="center")
for i, row in enumerate(rk.itertuples()):
    yi = yv[i]
    axD.text(700, yi, f"{int(row.rank_min)}\u2013{int(row.rank_max)}", fontsize=5.9,
             color=GROUP_COLOR[DRUG_GROUP[row.drug]], ha="left", va="center")

# legend placed under the axis (below the row of drug ticks) so it never
# overlaps the interval bars or the right-hand annotation column
axD.legend(handles=[
    Patch(facecolor=OI["vermillion"], label="positive bridge"),
    Patch(facecolor=OI["orange"], label="high-risk candidate"),
    Patch(facecolor=OI["blue"], label="negative control"),
], loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=5.8,
    frameon=False, handlelength=1.3, columnspacing=1.1)
pu.label_panel(axD, "D")


# ======================================================================
# Save
# ======================================================================
out_base = FIG / "Figure3_robustness"
pu.save_figure(fig, out_base, formats=("pdf", "png"), dpi=600)
plt.close(fig)
P(f"[saved] {out_base}.pdf / .png")

# ======================================================================
# Caption (written to out/S15_caption.txt for the manuscript)
# ======================================================================
caption = (
"**Figure 2.** Robustness of the risk ranking to modelling choices. "
"Results are from an independent re-implementation of the pipeline on a fixed "
"444/112 stratified split (Methods 2.9). "
"**A.** Five-fold cross-validated AUROC under the four label-encoding schemes "
"(L1 Most vs No, n = 556; L2 expanded-positive (Most + Less) vs No, n = 884; "
"L3 ordinal three-class, n = 884; L4 four-class folded, n = 556); the primary "
"two-class scheme (L1) and the four-class folded scheme (L4) perform comparably, "
"whereas the expanded-positive (L2) and ordinal (L3) schemes are weaker. "
"**B.** Five-fold cross-validated AUROC of the five descriptor spaces; the "
"Morgan fingerprint alone (F1) and its concatenation with the physicochemical "
"descriptors (F3) are the strongest, the physicochemical descriptors alone (F2) "
"are markedly weaker, and the 3D-containing spaces (F4, F5; hatched) give no gain "
"over 2D \u2014 adding the 3D descriptors to the 2D space (F5 vs F3) gives "
"\u0394AUROC = \u22120.015 [\u22120.024, \u22120.005], failing the pre-registered "
"threshold of \u0394AUROC \u2265 0.03 with an interval excluding zero. "
"**C.** Consensus layering of the 659 uncategorised compounds by how many of the "
"seven scoring configurations place them in the high-score tail (top 10%): "
"**(C1)** layer size (11 high-consensus, 205 discordant, 443 low-consensus) and "
"**(C2)** the distribution of the vote count $n_{high}$, which shows that no "
"compound is placed in the tail by all seven models. "
"**D.** Rank range of the prespecified validation panel across all 36 combinations "
"of label-encoding (4) \u00d7 classifier (3) \u00d7 descriptor space (3) of the "
"659-compound ranking; the high-risk candidates stay high and the negative "
"controls stay low, and the widest intervals (up to rank 598 for Cabotegravir) "
"arise from the alternative encodings and the weak physicochemical space rather "
"than from the primary configuration. "
"All panels are derived from an independent re-implementation using a fixed "
"444/112 stratified split; the rank intervals in **D** are therefore not "
"numerically comparable with the primary risk scores of Section 3.5."
)
cap_path = OUT / "S15_caption.txt"
cap_path.write_text(caption, encoding="utf-8")
P(f"[saved] {cap_path}")
P("done.")
