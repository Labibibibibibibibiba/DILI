# Interpretable machine learning with statistical fragment enrichment identifies DILI-associated chemical substructures

Code and derived data accompanying the manuscript submitted to *Chemical Research in
Toxicology* (ACS).

## Contents

| Directory | What it holds |
|---|---|
| `main_pipeline/` | Primary analysis pipeline — data curation, molecular featurization, model training/selection, SHAP attribution, full-family fragment enrichment with FDR control, structural backtracking, and the published figures. Scripts are numbered in execution order (`01_…` → `24_…`); `fig*.py` are the figure scripts. |
| `robustness/` | Independent re-implementation used for the sensitivity audit (label-encoding scheme × classifier × descriptor space; manuscript §2.8 / §3.5). `S16_supplementary_tables.py` re-assembles Tables S1–S7 of the Supplementary Information from the files below. |
| `results/` | Tabular outputs of the robustness analysis (CSV / JSON / logs) underlying Figures 3 and S1–S6, plus the machine-readable supplementary tables (`Supplementary_Table_S1.csv`, `Supplementary_Table_S2.csv`). |
| `data/` | Derived datasets and statistics underlying the main results: final feature matrix, train/test splits, cross-validation folds, model metrics, SHAP values, the full-family enrichment table (`enrichment_fullfamily_bitidentity.xlsx`), risk predictions for the uncategorized compounds, and the external-validation outputs. |
| `figures/` | The ten final figures (Figures 1–4 and Figures S1–S6), each as PNG (600 dpi) and PDF. |

## Data sources

- **DILIrank 2.0** — FDA reference list ranking drugs by human DILI concern. The copy used
  is included as `data/Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx`;
  it is also available directly from the U.S. Food and Drug Administration.
- **DILIst** — independent, FDA-derived drug list used for external validation
  (`data/external/`).
- **PubChem** — compound structures retrieved by name (SMILES).

This is a purely computational study on publicly available reference data; it involves no
human or animal subjects and required no ethics approval.

## Environment

Python **3.13.14**; package versions in `requirements.txt`. Random seeds used throughout:
data split 42, cross-validation 42, hyper-parameter search 42, scaffold split 42,
bootstrap 20261003.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

> The SHAP step (`main_pipeline/17_SHAP_analysis.py`, `main_pipeline/21_shap_enrichment_v2.py`)
> uses the `shap` package, pinned as `shap==0.52.0` in `requirements.txt`.

## Running the pipeline

Scripts in `main_pipeline/` are numbered in execution order and read/write a `data/`
directory relative to the working directory. **The scripts contain absolute paths from the
original analysis machine** (e.g. `D:/DILI/data`); adjust these to your local layout before
running. `robustness/` is self-contained and reads its inputs from `results/`.

## Reproducibility notes

- Enrichment statistics are reported at two distinct levels that must not be conflated:
  the **fingerprint-bit level** (one Fisher test per Morgan bit) and the **substructure
  level** (one test per `(bit, canonical SMILES)` pair over the full 809-fragment family,
  Benjamini–Hochberg controlled). Figures and text state which level applies.
- The `L4` label-encoding scheme reports a value obtained by maximising a pooled
  out-of-fold AUROC over a λ × μ weight grid, and therefore carries no cross-validated
  confidence interval; this is stated explicitly wherever the value appears.
- Three cross-validation scales appear in the manuscript (primary full-556 CV, primary
  train-444 CV, and the fixed 444/112 robustness split). They are not numerically
  comparable, and the manuscript declares which scale each reported figure uses.

## Use of generative AI

Per the manuscript's Acknowledgments: an AI assistant (WorkBuddy, powered by the
DeepSeek-V4.1-Flash large language model) was used to assist with code development,
figure generation, and language editing. All code was executed and its output verified by
the author, and every reported statistic was recomputed from the source data.

## License

Code: MIT (see `LICENSE`). Derived data and figures: CC BY 4.0.
