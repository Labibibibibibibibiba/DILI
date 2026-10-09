#!/usr/bin/env python3
"""
21_shap_enrichment_v2.py
========================
用 v2 模型重算 SHAP 与片段富集，并一次性解决审稿 M2（FDR 检验族披露）与 M6（OR 95% CI）。

输出（D:/DILI/data/）：
  shap_values_556_v2.npy / shap_feature_names_v2.json / shap_importance_v2.xlsx
  FP_enrichment_results_v2.xlsx        位点级（compound-level 2x2 + OR 95%CI）
  FP_structure_mapping_result_v2.xlsx  SHAP 重要位 → 子结构 SMILES 映射
  structure_enrichment_statistics_v2.xlsx  子结构级（限定族，BH FDR）
  substructure_enrichment_fullfamily.xlsx  子结构级（全指纹空间，count>=5，BH FDR）
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import shap
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

RDLogger.DisableLog("rdApp.*")

DATA = "D:/DILI/data"
MIN_COUNT = 5
TOP_K_FEATURES = 30          # SHAP 排序前若干特征中筛 FP 位
t0 = time.time()
LOG = []


def log(m=""):
    print(m, flush=True)
    LOG.append(str(m))


def or_ci(a, b, c, d):
    """OR + 95% CI（Woolf，任一方格为 0 时加 0.5 校正）"""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo, hi = np.exp(np.log(or_) - 1.96 * se), np.exp(np.log(or_) + 1.96 * se)
    return float(or_), float(lo), float(hi)


# =============================================================================
# Part 0 — 数据
# =============================================================================
df = pd.read_excel(f"{DATA}/DILI_ML_dataset_v5_final.xlsx")
X = df.drop(columns=["CompoundName", "SMILES", "DILI_label"])
y = df["DILI_label"].values
names = df["CompoundName"].values
smiles = df["SMILES"].values
log(f"[0] 556 数据集: {X.shape}  DILI+={int(y.sum())}  DILI-={int((y==0).sum())}")

model = joblib.load(f"{DATA}/best_RF_model_v2.pkl")
X_train = pd.read_excel(f"{DATA}/X_train.xlsx")

# =============================================================================
# Part 1 — SHAP（556 全数据集，probability + interventional）
# =============================================================================
log("\n[1] SHAP")
background = shap.sample(X_train, 100, random_state=7)
explainer = shap.TreeExplainer(
    model, data=background, feature_perturbation="interventional", model_output="probability"
)
exp = explainer(X)[..., 1]
recon = np.asarray(exp.base_values) + exp.values.sum(axis=1)
err = float(np.max(np.abs(recon - model.predict_proba(X)[:, 1])))
log(f"    加性验证 max|base+Σvalues − predict_proba| = {err:.3e}")

imp = np.abs(exp.values).mean(axis=0)
feat_names = list(exp.feature_names)
order = np.argsort(imp)[::-1]
imp_df = pd.DataFrame({"Feature": [feat_names[i] for i in order],
                       "MeanAbsSHAP": [float(imp[i]) for i in order]}).head(TOP_K_FEATURES)
log("    Top 12 特征:")
for i in range(12):
    log(f"      {i+1:2d}. {feat_names[order[i]]:16s} {imp[order[i]]:.6f}")

np.save(f"{DATA}/shap_values_556_v2.npy", np.asarray(exp.values))
json.dump(feat_names, open(f"{DATA}/shap_feature_names_v2.json", "w"))
imp_df.to_excel(f"{DATA}/shap_importance_v2.xlsx", index=False)

important_bits = [f for f in imp_df["Feature"] if str(f).startswith("FP_")]
important_bits = [int(b.split("_")[1]) for b in important_bits]
log(f"    SHAP Top{TOP_K_FEATURES} 中的 FP 位（{len(important_bits)} 个）: {['FP_%d'%b for b in important_bits]}")

# =============================================================================
# Part 2 — 位点级富集（compound-level 2x2 + OR/CI/p）
# =============================================================================
log("\n[2] 位点级富集")
# 一次性算出每个化合物的 bitInfo
bitinfo_all = []
for smi in smiles:
    m = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
    bi = {}
    if m is not None:
        AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, bitInfo=bi)
    bitinfo_all.append(bi)
log(f"    解析成功 {sum(1 for b in bitinfo_all if b)} / {len(bitinfo_all)}")

rows = []
for bit in important_bits:
    has = np.array([1 if bit in bi else 0 for bi in bitinfo_all])
    a = int(((has == 1) & (y == 1)).sum())
    b = int(((has == 1) & (y == 0)).sum())
    c = int(((has == 0) & (y == 1)).sum())
    d = int(((has == 0) & (y == 0)).sum())
    or_, lo, hi = or_ci(a, b, c, d)
    _, p = fisher_exact([[a, b], [c, d]])
    rows.append({"Feature": f"FP_{bit}", "carriers": a + b,
                 "FP_DILI_positive": a, "FP_DILI_negative": b,
                 "NoFP_DILI_positive": c, "NoFP_DILI_negative": d,
                 "DILI_rate_carriers": a / (a + b) if (a + b) else np.nan,
                 "OddsRatio": or_, "OR_CI_low": lo, "OR_CI_high": hi, "p_value": p})
fp_df = pd.DataFrame(rows).sort_values("OddsRatio", ascending=False)
fp_df.to_excel(f"{DATA}/FP_enrichment_results_v2.xlsx", index=False)
log(fp_df.to_string(index=False))

# =============================================================================
# Part 3 — SHAP 重要位 → 子结构 SMILES 映射
# =============================================================================
log("\n[3] 子结构映射（限定族：SHAP 重要位）")
map_rows = []
for i, (name, smi, lab) in enumerate(zip(names, smiles, y)):
    m = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
    if m is None:
        continue
    for bit, locations in bitinfo_all[i].items():
        if bit not in important_bits:
            continue
        for atom_id, radius in locations:
            env = Chem.FindAtomEnvironmentOfRadiusN(m, radius, atom_id)
            if not len(env):
                continue
            sub = Chem.MolToSmiles(Chem.PathToSubmol(m, env))
            map_rows.append({"CompoundName": name, "DILI_label": int(lab),
                             "Fingerprint": f"FP_{bit}", "Atom_ID": atom_id,
                             "Radius": radius, "Substructure_SMILES": sub})
map_df = pd.DataFrame(map_rows)
map_df.to_excel(f"{DATA}/FP_structure_mapping_result_v2.xlsx", index=False)
log(f"    映射条目 {map_df.shape}")


def substructure_enrichment(mapping_df, tag):
    """子结构级富集：以化合物为单位（去重），对 (Fingerprint, Substructure) 分组"""
    d = mapping_df.dropna(subset=["Substructure_SMILES"]).drop_duplicates(
        subset=["CompoundName", "Fingerprint", "Substructure_SMILES"])
    res = []
    for (fp, sub), g in d.groupby(["Fingerprint", "Substructure_SMILES"]):
        carriers = set(g["CompoundName"])
        cnt = len(carriers)
        if cnt < MIN_COUNT:
            continue
        pos = int((g["DILI_label"] == 1).sum())
        neg = int((g["DILI_label"] == 0).sum())
        out = ~names.isin(carriers) if hasattr(names, "isin") else ~pd.Series(names).isin(carriers).values
        opos = int(y[out].sum())
        oneg = int((y[out] == 0).sum())
        or_, lo, hi = or_ci(pos, neg, opos, oneg)
        _, p = fisher_exact([[pos, neg], [opos, oneg]])
        res.append({"Fingerprint": fp, "Substructure_SMILES": sub, "Count": cnt,
                    "DILI_positive": pos, "DILI_negative": neg,
                    "DILI_rate": pos / cnt, "OddsRatio": or_,
                    "OR_CI_low": lo, "OR_CI_high": hi, "p_value": p})
    r = pd.DataFrame(res)
    if len(r) == 0:
        return r
    rej, fdr, _, _ = multipletests(r["p_value"], method="fdr_bh")
    r["FDR"] = fdr
    r["Significant"] = rej
    r["Family_tag"] = tag
    r["Family_size"] = len(r)
    return r.sort_values(["FDR", "OddsRatio"], ascending=[True, False])


log("\n[4] 子结构级富集（限定族）")
sub_restricted = substructure_enrichment(map_df, "SHAP-important bits")
log(f"    检验族大小 n = {len(sub_restricted)}")
log(sub_restricted.to_string(index=False))
sub_restricted.to_excel(f"{DATA}/structure_enrichment_statistics_v2.xlsx", index=False)

# =============================================================================
# Part 5 — 全指纹空间子结构富集（强化 M2：把「弱点」变「亮点」）
# =============================================================================
log("\n[5] 全指纹空间子结构富集（所有位点，按子结构 SMILES 合并）")
full_rows = []
for i, (name, smi, lab) in enumerate(zip(names, smiles, y)):
    m = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
    if m is None:
        continue
    seen = set()
    for bit, locations in bitinfo_all[i].items():
        for atom_id, radius in locations:
            env = Chem.FindAtomEnvironmentOfRadiusN(m, radius, atom_id)
            if not len(env):
                continue
            s = Chem.MolToSmiles(Chem.PathToSubmol(m, env))
            if s in seen:
                continue
            seen.add(s)
            full_rows.append({"CompoundName": name, "DILI_label": int(lab),
                              "Fingerprint": f"FP_{bit}", "Substructure_SMILES": s})
full_df = pd.DataFrame(full_rows)
log(f"    全空间映射条目 {full_df.shape}")

# 按子结构 SMILES 合并（同一子结构可能来自多个指纹位 —— 回应 M8 的合并规则）
agg = full_df.drop_duplicates(subset=["CompoundName", "Substructure_SMILES"])
rows = []
for sub, g in agg.groupby("Substructure_SMILES"):
    carriers = set(g["CompoundName"])
    cnt = len(carriers)
    if cnt < MIN_COUNT:
        continue
    pos = int((g["DILI_label"] == 1).sum())
    neg = int((g["DILI_label"] == 0).sum())
    out = ~pd.Series(names).isin(carriers).values
    opos, oneg = int(y[out].sum()), int((y[out] == 0).sum())
    or_, lo, hi = or_ci(pos, neg, opos, oneg)
    _, p = fisher_exact([[pos, neg], [opos, oneg]])
    rows.append({"Substructure_SMILES": sub, "Count": cnt, "DILI_positive": pos,
                 "DILI_negative": neg, "DILI_rate": pos / cnt, "OddsRatio": or_,
                 "OR_CI_low": lo, "OR_CI_high": hi, "p_value": p})
full_stat = pd.DataFrame(rows)
rej, fdr, _, _ = multipletests(full_stat["p_value"], method="fdr_bh")
full_stat["FDR"] = fdr
full_stat["Significant"] = rej
full_stat = full_stat.sort_values(["FDR", "OddsRatio"], ascending=[True, False])
full_stat.to_excel(f"{DATA}/substructure_enrichment_fullfamily.xlsx", index=False)
log(f"    全空间检验族大小 n = {len(full_stat)}")
log(f"    FDR 显著且 OR>1 的子结构数 = {int((full_stat['Significant'] & (full_stat['OddsRatio']>1)).sum())}")
log(f"    FDR 显著且 OR<1 的子结构数 = {int((full_stat['Significant'] & (full_stat['OddsRatio']<1)).sum())}")
cols = ["Substructure_SMILES", "Count", "DILI_rate", "OddsRatio", "OR_CI_low", "OR_CI_high", "p_value", "FDR", "Significant"]
log("\n    全空间 FDR 最显著的前 10 个正富集子结构:")
log(full_stat[full_stat["OddsRatio"] > 1][cols].head(10).to_string(index=False))
log("\n    关键子结构查核:")
for key in ["cc(c)C", "CN"]:
    hit = full_stat[full_stat["Substructure_SMILES"] == key]
    if len(hit):
        log("      " + hit[cols].to_string(index=False).replace("\n", "\n      "))
    else:
        log(f"      {key}: 未进入 count>=5 的检验族")

summary = {
    "shap_additivity_max_err": err,
    "shap_top_features": imp_df.head(10).to_dict("records"),
    "important_bits": ["FP_%d" % b for b in important_bits],
    "restricted_family_size": int(len(sub_restricted)),
    "full_family_size": int(len(full_stat)),
    "full_family_sig_pos": int((full_stat["Significant"] & (full_stat["OddsRatio"] > 1)).sum()),
    "full_family_sig_neg": int((full_stat["Significant"] & (full_stat["OddsRatio"] < 1)).sum()),
}
json.dump(summary, open(f"{DATA}/enrichment_summary_v2.json", "w"), indent=2, ensure_ascii=False, default=str)
open(f"{DATA}/21_shap_enrichment_log.txt", "w", encoding="utf-8").write("\n".join(LOG))
log(f"\n[6] 完成，耗时 {time.time()-t0:.0f}s")
