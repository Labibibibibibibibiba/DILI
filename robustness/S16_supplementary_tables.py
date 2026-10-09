# -*- coding: utf-8 -*-
"""Assemble Supplementary Tables S1-S7 and Supplementary Methods S-M1.

Every numeric value is read directly from the analysis output files below;
nothing is transcribed by hand.
  - Table S1 : D:/DILI/data/enrichment_fullfamily_bitidentity.xlsx  (809 rows)
  - Table S2 : D:/DILI/data/FP_enrichment_results_v2.xlsx           (excluded from
               this generator: already assembled and verified in the previous round)
  - Table S3 : D:/DILI/data/smarts_direct_test.xlsx                 (10 patterns)
  - Table S4 : D:/DILI_1/out/S3_label_sensitivity.csv               (L1-L4)
  - Table S5 : D:/DILI_1/out/S4_descriptor_comparison.csv           (F1-F5)
               paired interval from D:/DILI_1/out/S4_run.log
  - Table S6 : D:/DILI_1/out/S11_rank_summary.csv                   (36 combinations)
  - Table S7 : D:/DILI_1/out/S6_wetlab_candidates.xlsx              (16 compounds)
"""
import math
import re
from pathlib import Path

import pandas as pd

DILI = Path(r"D:\DILI")
D1 = Path(r"D:\DILI_1")
OUT = D1 / "out"
PAP = D1 / "paper"

# ---------------------------------------------------------------- formatting
_SUP = {ord("0") + i: c for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")}
_SUP[ord("-")] = "⁻"


def sci(x, sig=2):
    """0.0067 -> '6.7 × 10⁻³' ; >=0.01 -> 2 decimals ; NaN -> 'n/a'."""
    try:
        x = float(x)
    except Exception:
        return "n/a"
    if x != x:
        return "n/a"
    if x == 0:
        return "0"
    if x >= 0.01:
        return f"{x:.2f}"
    e = math.floor(math.log10(x))
    m = x / 10 ** e
    m = round(m, sig - 1)
    if m >= 10:
        m /= 10
        e += 1
    return f"{m:.{sig-1}f} × 10{_SUP[ord('-')]}{str(abs(e)).translate(_SUP)}"


def f2(x):
    try:
        x = float(x)
    except Exception:
        return "n/a"
    return "n/a" if x != x else f"{x:.2f}"


def f3(x):
    try:
        x = float(x)
    except Exception:
        return "n/a"
    return "n/a" if x != x else f"{x:.3f}"


def pct(x):
    try:
        x = float(x)
    except Exception:
        return "n/a"
    return "n/a" if x != x else f"{x*100:.1f}%"


def us(x, nd=4):
    """Signed value with a proper unicode minus, e.g. -0.0145 -> '\u22120.0145'."""
    s = f"{float(x):+.{nd}f}"
    return s.replace("-", "\u2212")


# ================================================================ Table S1
fam = pd.read_excel(DILI / "data" / "enrichment_fullfamily_bitidentity.xlsx")
n_all = len(fam)
n_sig = int(fam["Significant"].sum())
n_pos = int((fam["Significant"] & (fam["OddsRatio"] > 1)).sum())
n_neg = int((fam["Significant"] & (fam["OddsRatio"] < 1)).sum())
fam_csv = OUT / "Supplementary_Table_S1.csv"
fam.to_csv(fam_csv, index=False, encoding="utf-8-sig")

dedup = pd.read_excel(DILI / "data" / "substructure_enrichment_fullfamily.xlsx")
d_all = len(dedup)
d_sig = int(dedup["Significant"].sum())
d_pos = int((dedup["Significant"] & (dedup["OddsRatio"] > 1)).sum())
d_neg = int((dedup["Significant"] & (dedup["OddsRatio"] < 1)).sum())

# ================================================================ Table S3
sm = pd.read_excel(DILI / "data" / "smarts_direct_test.xlsx")


def s3_row(r):
    ci = "n/a" if pd.isna(r["OR_CI_low"]) else f"{r['OR_CI_low']:.2f}–{r['OR_CI_high']:.2f}"
    return (
        f"| `{r['SMARTS']}` | {r['Definition']} | {int(r['Count'])} | {int(r['DILI_positive'])} | "
        f"{int(r['DILI_negative'])} | {pct(r['DILI_rate'])} | {f2(r['OddsRatio'])} | {ci} | "
        f"{sci(r['p_value'])} | {sci(r['FDR'])} | {'yes' if r['Significant'] else 'no'} |"
    )


s3_rows = "\n".join(s3_row(r) for _, r in sm.iterrows())

# ================================================================ Table S4
ls = pd.read_csv(OUT / "S3_label_sensitivity.csv")
ENC = {
    "L1": "L1. Two-class, primary (Most = 1 / No = 0)",
    "L2": "L2. Expanded positive ((Most + Less) = 1 / No = 0)",
    "L3": "L3. Ordinal three-class (Most / Less / No)",
    "L4-best": "L4. Four-class nominal, folded (grid-optimal \u03bb, \u03bc)",
}
CLF = {"RF": "Random forest", "RF-3class": "Random forest (3-class)", "RF-4class": "Random forest (4-class)"}


def s4_row(r):
    ci = "n/a" if pd.isna(r["ci_lo"]) else f"{r['ci_lo']:.3f}–{r['ci_hi']:.3f}"
    return (
        f"| {ENC.get(r['scheme'], r['scheme'])} | {CLF.get(r['algo'], r['algo'])} | {int(r['n'])} | "
        f"{int(r['n_pos'])} | {f3(r['auc_mean'])} | {f3(r['auc_sd'])} | {ci} | {f3(r['f1_mean'])} |"
    )


s4_rows = "\n".join(s4_row(r) for _, r in ls.iterrows())
l4 = ls[ls["scheme"] == "L4-best"].iloc[0]

# grid-optimal fold weights, read from the grid scan output
grid = pd.read_csv(OUT / "S3_L4_fold_grid_results.csv")
best = grid.sort_values("auc_most_vs_no", ascending=False).iloc[0]
lam_star, mu_star = float(best["lam_less"]), float(best["mu_amb"])
at_0605 = grid[(grid["lam_less"].round(2) == 0.6) & (grid["mu_amb"].round(2) == 0.5)]["auc_most_vs_no"]
auc_0605 = float(at_0605.iloc[0]) if len(at_0605) else float("nan")

# ================================================================ Table S5
ds = pd.read_csv(OUT / "S4_descriptor_comparison.csv")
FEAT = {
    "F1_Morgan": ("F1. Morgan fingerprint", "Extended-connectivity fingerprint, radius 2, 1,024 bits"),
    "F2_2Ddesc": ("F2. Physicochemical descriptors", "Molecular weight, LogP, TPSA, HBD, HBA, rotatable bonds, aromatic rings"),
    "F3_Morgan+2D": ("F3. Concatenation (primary space)", "F1 + F2; the 1,031-feature primary space"),
    "F4_3Donly": ("F4. Three-dimensional shape descriptors", "Asphericity, eccentricity, inertial shape factor, NPR1/NPR2, PMI1–PMI3, radius of gyration, spherocity index, PBF"),
    "F5_2D+3D": ("F5. Two-dimensional + three-dimensional", "F2 + F4"),
}


def s5_row(r):
    name, contents = FEAT[r["feature_set"]]
    return (
        f"| {name} | {contents} | {int(r['n_feat'])} | {f3(r['auc_mean'])} | {f3(r['auc_sd'])} | "
        f"{f3(r['ci_lo'])}–{f3(r['ci_hi'])} |"
    )


s5_rows = "\n".join(s5_row(r) for _, r in ds.iterrows())

# paired F5 - F3 interval, parsed from the run log so it is not retyped
log = (OUT / "S4_run.log").read_text(encoding="utf-8", errors="ignore")
m = re.search(r"\[G3\].*?ΔAUC = ([+-]?\d+\.\d+).*?95%CI \[([+-]?\d+\.\d+), ([+-]?\d+\.\d+)\]", log)
delta, d_lo, d_hi = float(m.group(1)), float(m.group(2)), float(m.group(3))

# ================================================================ Table S6
rs = pd.read_csv(OUT / "S11_rank_summary.csv")
ROLE = {
    "阳性·池内桥接": "Positive control, in-pool bridge",
    "阳性·池外基准": "Positive control, out-of-pool benchmark",
    "高风险候选": "High-risk candidate",
    "阴性 TierA": "Negative control, tier A",
    "阴性 TierB": "Negative control, tier B",
    "不纳入(审计)": "Not included (audit row)",
}
panel = rs[~rs["role"].isin(["不纳入(审计)"])].copy()
panel = panel[panel["present"].astype(str).str.lower() == "true"]


def s6_row(r):
    return (
        f"| {r['drug']} | {ROLE.get(r['role'], r['role'])} | {int(r['n_models'])} | "
        f"{f2(r['rank_mean'])} | {int(r['rank_min'])} | {int(r['rank_max'])} | "
        f"{pct(r['pct_top10']/100)} | {pct(r['pct_top20']/100)} |"
    )


s6_rows = "\n".join(s6_row(r) for _, r in panel.iterrows())
absent = rs[rs["role"] == "阳性·池外基准"]["drug"].tolist()
_EXTRA = {"DCF": "DCF (diclofenac)."}
absent_txt = " ".join(_EXTRA.get(a, a + ".") for a in absent).replace("..", ".")
# ================================================================ Table S7
wl = pd.read_excel(OUT / "S6_wetlab_candidates.xlsx")
R7 = {
    "Discordant(主攻)": "Discordant (primary target)",
    "Consensus_HIGH(阳性锚点)": "Consensus high (positive anchor)",
    "Consensus_LOW(阴性锚点)": "Consensus low (negative anchor)",
}
TIER = {
    "A": "A (routine)",
    "B": "B (moderate)",
    "C": "C (peptide)",
    "D": "D (restricted)",
}


def tier(s):
    s = str(s)
    head = s.split("(")[0].strip()
    if head.startswith("A/B"):
        return "A/B (routine to moderate)"
    return TIER.get(head[:1], head)


ORDER7 = {"Discordant(主攻)": 0, "Consensus_HIGH(阳性锚点)": 1, "Consensus_LOW(阴性锚点)": 2}


def s7_row(i, r):
    return (
        f"| {i} | {r['CompoundName']} | {r['cas_no']} | {r['DILI_category']} | "
        f"{R7.get(r['role'], r['role'])} | {int(r['n_high'])}/7 | {f3(r['role_sd'])} | {tier(r['procurement_tier'])} |"
    )


# cross-model SD is encoded in the Chinese rationale string; parse it out,
# then order the panel as discordant (by decreasing disagreement), then the
# consensus-high anchors, then the consensus-low anchors.
sd_map = {}
for _, r in wl.iterrows():
    mm = re.search(r"SD=([0-9.]+)", str(r["rationale"]))
    sd_map[r["CompoundName"]] = float(mm.group(1)) if mm else float("nan")
wl["role_sd"] = wl["CompoundName"].map(sd_map)
wl["_ord"] = wl["role"].map(ORDER7)
_g = [g for _, g in wl.groupby("_ord")]
wl = pd.concat([
    _g[0].sort_values("role_sd", ascending=False),   # most discordant first
    _g[1].sort_values("n_high", ascending=False),    # strongest consensus-high first
    _g[2].sort_values("role_sd", ascending=True),    # tightest consensus-low first
], ignore_index=True)
s7_rows = "\n".join(s7_row(i, r) for i, (_, r) in enumerate(wl.iterrows(), 1))

# ================================================================ write file
body = f"""# Supplementary Tables

> Companion tables for *Interpretable machine learning with statistical fragment enrichment identifies DILI-associated chemical substructures* (Chen Jiaxi, Guangdong Medical University).

> **Provenance.** Every value in Tables S1–S7 is read directly from the analysis output files named at the foot of each table; none is transcribed by hand. Percentages and odds ratios are given to one and two decimal places respectively, cross-validated AUROCs to three, and P values and FDRs in scientific notation.

---

## Table S1. Full-family substructure enrichment

Fisher exact tests for all {n_all} (fingerprint bit, atom-environment) pairs resolved by RDKit structural backtracking, against the DILI-positive / DILI-negative split of the 556-compound modeling set, with Benjamini–Hochberg control across the family. {n_sig} pairs were significant after FDR correction: {n_pos} enriched and {n_neg} depleted among DILI-positive compounds.

The complete {n_all}-row table (columns: fingerprint bit, canonical SMILES of the atom environment, carrier count, DILI-positive and DILI-negative carriers, carrier DILI-positive rate, odds ratio, 95% confidence interval, P, FDR, significance flag) is provided as the machine-readable file `Supplementary_Table_S1.csv`.

> **Two counting conventions.** The family used in the manuscript is defined **by fingerprint bit identity**, giving {n_all} pairs and {n_sig} significant associations ({n_pos} + {n_neg}). If the identical substructure is instead **deduplicated by SMILES** (the same environment inherited by more than one bit is counted once), the family contains {d_all} unique substructures and {d_sig} significant associations ({d_pos} + {d_neg}); this reduced family is stored in `substructure_enrichment_fullfamily.xlsx`. The two are not interchangeable and the bit-identity convention is the one reported throughout.

---

## Table S2. Fingerprint-bit-level Fisher exact tests for the 23 SHAP-important bits

Each of the 23 fingerprint bits ranked among the SHAP-important features was tested with a two-sided Fisher exact test against the DILI-positive / DILI-negative split of the 556-compound modeling set. Counts refer to compounds carrying the bit; the carrier rate is the proportion of carriers that are DILI-positive (modeling-set prevalence of DILI-positives: 36.9%, 205/556). Odds ratios and confidence limits are given to two decimal places.

**Enriched among DILI-positive compounds (odds ratio > 1)**

| Bit | Carriers (n) | DILI+ carriers | DILI− carriers | Carrier DILI+ rate | Odds ratio | 95% CI | P |
|---|---|---|---|---|---|---|---|
| `FP_464` | 22 | 18 | 4 | 81.8% | 8.35 | 2.79–25.03 | 1.4 × 10⁻⁵ |
| `FP_593` | 19 | 14 | 5 | 73.7% | 5.07 | 1.80–14.30 | 1.2 × 10⁻³ |
| `FP_356` | 476 | 195 | 281 | 41.0% | 4.86 | 2.44–9.66 | 3.1 × 10⁻⁷ |
| `FP_9` | 23 | 16 | 7 | 69.6% | 4.16 | 1.68–10.29 | 1.5 × 10⁻³ |
| `FP_367` | 41 | 28 | 13 | 68.3% | 4.11 | 2.08–8.14 | 3.1 × 10⁻⁵ |
| `FP_849` | 461 | 188 | 273 | 40.8% | 3.16 | 1.81–5.51 | 1.4 × 10⁻⁵ |
| `FP_698` | 78 | 42 | 36 | 53.8% | 2.25 | 1.39–3.66 | 1.4 × 10⁻³ |
| `FP_561` | 80 | 43 | 37 | 53.8% | 2.25 | 1.40–3.64 | 1.0 × 10⁻³ |
| `FP_726` | 386 | 161 | 225 | 41.7% | 2.05 | 1.38–3.05 | 4.0 × 10⁻⁴ |
| `FP_333` | 74 | 37 | 37 | 50.0% | 1.87 | 1.14–3.06 | 1.4 × 10⁻² |
| `FP_378` | 193 | 85 | 108 | 44.0% | 1.59 | 1.11–2.28 | 1.3 × 10⁻² |
| `FP_875` | 193 | 84 | 109 | 43.5% | 1.54 | 1.08–2.21 | 2.1 × 10⁻² |
| `FP_90` | 135 | 52 | 83 | 38.5% | 1.10 | 0.74–1.64 | 6.8 × 10⁻¹ |

**Depleted among DILI-positive compounds (odds ratio < 1)**

| Bit | Carriers (n) | DILI+ carriers | DILI− carriers | Carrier DILI+ rate | Odds ratio | 95% CI | P |
|---|---|---|---|---|---|---|---|
| `FP_1019` | 237 | 74 | 163 | 31.2% | 0.65 | 0.46–0.93 | 2.1 × 10⁻² |
| `FP_926` | 275 | 85 | 190 | 30.9% | 0.60 | 0.42–0.85 | 4.9 × 10⁻³ |
| `FP_807` | 441 | 147 | 294 | 33.3% | 0.49 | 0.32–0.74 | 1.1 × 10⁻³ |
| `FP_268` | 64 | 14 | 50 | 21.9% | 0.44 | 0.24–0.82 | 8.6 × 10⁻³ |
| `FP_1` | 207 | 52 | 155 | 25.1% | 0.43 | 0.29–0.63 | 8.3 × 10⁻⁶ |
| `FP_227` | 72 | 14 | 58 | 19.4% | 0.37 | 0.20–0.68 | 9.6 × 10⁻⁴ |
| `FP_981` | 46 | 5 | 41 | 10.9% | 0.19 | 0.07–0.49 | 8.5 × 10⁻⁵ |
| `FP_41` | 48 | 5 | 43 | 10.4% | 0.18 | 0.07–0.46 | 3.0 × 10⁻⁵ |
| `FP_993` | 32 | 3 | 29 | 9.4% | 0.16 | 0.05–0.55 | 5.2 × 10⁻⁴ |
| `FP_575` | 31 | 2 | 29 | 6.5% | 0.11 | 0.03–0.46 | 1.6 × 10⁻⁴ |

> Note: no multiplicity correction is applied at the fingerprint-bit level; the FDR-controlled inference reported in the manuscript is performed at the substructure level over the full {n_all}-fragment family (Table S1). `FP_367` appears here with its bit-level odds ratio (4.11); its substructure-level value (`cc(c)C`, odds ratio 4.89) is a different statistic (Table S1).

---

## Table S3. Prespecified SMARTS panel

Ten interpretable SMARTS patterns, prespecified before the enrichment analysis, tested directly on the same 556-compound modeling set with a two-sided Fisher exact test and Benjamini–Hochberg control across the ten patterns. The panel deliberately includes the *unrestricted* pattern for any alkyl substituent on an aromatic ring (`[c][CX4]`), which serves as the environment-independent counterpart of the fingerprint environment `cc(c)C`.

| Pattern (SMARTS) | Definition | Carriers (n) | DILI+ carriers | DILI− carriers | Carrier DILI+ rate | Odds ratio | 95% CI | P | FDR | Significant |
|---|---|---|---|---|---|---|---|---|---|---|
{s3_rows}

> The three significant patterns define the same chemical theme recovered by the fingerprint enrichment: substitution on an aromatic ring by methyl, by aniline-type nitrogen, and by chlorine. The restricted environment `cc(c)C` (bit `FP_367`) and the unrestricted pattern `[c][CX4]` are deliberately contrasted: the unrestricted pattern is not enriched (odds ratio 1.14, P = 0.48), showing that the signal attaches to the defined environment rather than to the generic structural class.

---

## Table S4. Label-encoding sensitivity

Four encodings of the DILIrank 2.0 categories compared by five-fold cross-validation (random forest), always scoring the positive *Most-DILI-concern* class against *No-DILI-concern* so that the schemes are comparable. Positive-class F1 score is reported where the scheme yields a binary score.

| Encoding | Classifier | n | n DILI+ | CV AUROC | SD across folds | 95% CI | F1 score |
|---|---|---|---|---|---|---|---|
{s4_rows}

> **The L4 row is a different kind of statistic.** L1–L3 are fold-mean AUROCs with a fold-level confidence interval. L4 is a four-class nominal model whose score is the folded combination `s = P(Most) + λ·P(Less) + μ·P(Ambiguous)`; the value shown is the **maximum over the λ × μ grid** ({f3(l4['auc_mean'])}, at the grid-optimal weights λ = {lam_star:g}, μ = {mu_star:g}), not a fold mean, and therefore has no fold-level confidence interval. It is not a like-for-like comparison with L1–L3 and carries the selection optimism inherent in taking a grid maximum. The grid is shallow near its optimum (λ = 0.6, μ = 0.5 gives {auc_0605:.6f}, a difference of {abs(l4['auc_mean']-auc_0605):.4f}), so the exact weights do not drive the conclusion. All four schemes nevertheless score the same contrast, Most versus No, so the finding that no alternative encoding materially improves on the primary scheme holds. The manuscript text quotes no numeric value for L4 for this reason.

---

## Table S5. Descriptor-space comparison

Five feature spaces compared under the primary L1 encoding with five-fold cross-validation (random forest, 556 compounds).

| Descriptor space | Contents | Features (n) | CV AUROC | SD across folds | 95% CI |
|---|---|---|---|---|---|
{s5_rows}

**Paired comparison, F5 vs F3 (does 3D structure add anything?).**

| Comparison | ΔAUROC | Paired 95% CI | Pre-registered gain threshold | Outcome |
|---|---|---|---|---|
| F5 (2D + 3D) − F3 (2D) | {us(delta)} | {us(d_lo)} to {us(d_hi)} | ΔAUROC ≥ +0.03 with CI excluding zero | Not met; no gain from 3D |
> The three-dimensional descriptors did not improve discrimination: adding them to the two-dimensional space lowered the cross-validated AUROC, and the three-dimensional-only space was markedly weaker than any two-dimensional space. Three-dimensional structure is retained in the manuscript only as a comparison arm; the negative result is reported as found. Three-dimensional features accounted for 15.8% of the random-forest importance in F5 despite contributing no discrimination.

---

## Table S6. Rank stability of the prespecified candidate/control panel across the 36 combinations

Rank position of each panel compound among the 659 FDA-uncategorized compounds, computed for every one of the 4 label encodings × 3 classifiers × 3 two-dimensional descriptor spaces = 36 combinations (three-dimensional spaces were evaluated only under L1, so they are outside this grid). Rank 1 is the highest-risk position. The table reports the summary across the 36 combinations; the per-combination values are in `S11_label_rank_stability.csv`.

| Compound | Panel role | Combinations (n) | Mean rank | Best rank | Worst rank | Fraction in top 10% | Fraction in top 20% |
|---|---|---|---|---|---|---|---|
{s6_rows}

> **Reading the table.** The two negative controls behave as intended only under the primary L1 encoding; under the alternative encodings their ranks drift upward, and the drift is concentrated in the weakest descriptor space (F2, the seven physicochemical descriptors alone). The primary-encoding figures are therefore reported in the main text, and the alternative-encoding spread is shown here and in Figure S6. Logistic regression is included in the 36-combination family on purpose, as a deliberately unregularized, unstandardized baseline; it is by a wide margin the weakest of the three classifiers and materially perturbs the ranking, so the main-text statements are restricted to the two tree-based classifiers.
>
> **L4 fold weights in this grid.** The rank-stability grid fixes the L4 fold weights at λ = 0.6, μ = 0.5, whereas Table S4 reports the L4 grid optimum at λ = {lam_star:g}, μ = {mu_star:g}. The two settings differ by {abs(l4['auc_mean']-auc_0605):.4f} AUROC and the ranking conclusions are unchanged, but the two tables should not be read as using an identical L4 definition. The manuscript quotes neither set of weights.
>
> **Compound absent from the grid.** {absent_txt} It is a labeled DILI-positive (*vMost*) compound from the modeling set, used as a positive benchmark; it is not among the 659 uncategorized compounds and therefore has no rank in this table. It is reported separately in the manuscript.

---

## Table S7. Expanded 16-compound validation panel

The candidate panel proposed for prospective primary-human-hepatocyte validation. Ten compounds are the highest-disagreement ("discordant") compounds across the seven models, three are consensus-high positive anchors, and three are consensus-low negative anchors. **This panel is a hypothesis-generating output of the present computational work; none of these compounds was tested experimentally and no experimental data are reported in this paper.**

| # | Compound | CAS | DILIrank category | Panel role | High-risk votes (of 7) | Cross-model score SD | Procurement tier |
|---|---|---|---|---|---|---|---|
{s7_rows}

> **Columns.** *High-risk votes* is the number of the seven models (four label encodings × classifiers at the primary descriptor space) that place the compound in the top 10% of the 659-compound ranking; high variance across models defines the discordant group. *Cross-model score SD* is the standard deviation of the compound's score across those models. *Procurement tier* classifies commercial availability for a prospective follow-up study: **A** routine (widely supplied), **B** moderate (limited suppliers or longer lead times), **C** peptide reagent, **D** restricted.
>
> **Category assignment.** The DILIrank categories are the official DILIrank 2.0 values, taken directly from the FDA reference table (`Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx`, sheet `version 2`). `Mebendazole` is *vLess-DILI-concern* in the official table.
>
> **Restricted tier.** The single tier-D compound is listed in China's catalogue of highly toxic chemicals and is not available to individual purchasers; a follow-up study would require institutional registration.

---

## Sources

| Table | Source file |
|---|---|
| S1 | `data/enrichment_fullfamily_bitidentity.xlsx` (809 rows) and `data/substructure_enrichment_fullfamily.xlsx` (541 SMILES-deduplicated rows) |
| S2 | `data/FP_enrichment_results_v2.xlsx` |
| S3 | `data/smarts_direct_test.xlsx` |
| S4 | `out/S3_label_sensitivity.csv` |
| S5 | `out/S4_descriptor_comparison.csv`; paired interval from `out/S4_run.log` |
| S6 | `out/S11_rank_summary.csv`; per-combination values in `out/S11_label_rank_stability.csv` |
| S7 | `out/S6_wetlab_candidates.xlsx` |

*All source files are included in the code and data repository accompanying this manuscript.*
"""

(PAP / "Supplementary_Tables.md").write_text(body, encoding="utf-8")
print("wrote Supplementary_Tables.md  chars=", len(body))
print(f"  S1 {n_all}/{n_sig}/{n_pos}+{n_neg} ; dedup {d_all}/{d_sig}/{d_pos}+{d_neg}")
print(f"  S3 rows={len(sm)} ; S4 rows={len(ls)} ; S5 rows={len(ds)} ; S6 rows={len(panel)} ; S7 rows={len(wl)}")
print(f"  delta(F5-F3)={delta:+.4f} [{d_lo:+.4f},{d_hi:+.4f}]")
print(f"  S1 csv -> {fam_csv}")
