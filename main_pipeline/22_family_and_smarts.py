#!/usr/bin/env python3
"""
22_family_and_smarts.py
=======================
回应审稿 M2（FDR 检验族）与 M8（同一子结构来自多个指纹位的合并规则）。

做三件事：
  (1) 全指纹空间富集，**保留指纹位身份**（fragment := Morgan bit），count>=5 → BH FDR；
      报告 Familiesize 与关键位点在该族中的位置。
  (2) 直接 SMARTS 验证：把「烷基取代芳香环」等警示定义成 SMARTS，在 556 化合物上
      直接做 compound-level Fisher 检验（不依赖指纹哈希），给出 OR/95%CI/p/FDR。
  (3) 对照：把同一 SMARTS 在「仅 FP_367 携带者」与「全体携带者」两种口径下比较，
      量化「环境特异性 vs 裸 SMARTS」的差别（诚实性所需）。
"""

import json

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

RDLogger.DisableLog("rdApp.*")
DATA = "D:/DILI/data"
MIN_COUNT = 5
LOG = []


def log(m=""):
    print(m, flush=True)
    LOG.append(str(m))


def or_ci(a, b, c, d):
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    o = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(o), float(np.exp(np.log(o) - 1.96 * se)), float(np.exp(np.log(o) + 1.96 * se))


def test_carriers(carrier_mask, y, label, extra=None):
    a = int((carrier_mask & (y == 1)).sum())
    b = int((carrier_mask & (y == 0)).sum())
    c = int((~carrier_mask & (y == 1)).sum())
    d = int((~carrier_mask & (y == 0)).sum())
    o, lo, hi = or_ci(a, b, c, d)
    _, p = fisher_exact([[a, b], [c, d]])
    row = {"Definition": label, "Count": a + b, "DILI_positive": a, "DILI_negative": b,
           "DILI_rate": a / (a + b) if (a + b) else np.nan,
           "OddsRatio": o, "OR_CI_low": lo, "OR_CI_high": hi, "p_value": p}
    if extra:
        row.update(extra)
    return row


# =============================================================================
df = pd.read_excel(f"{DATA}/DILI_ML_dataset_v5_final.xlsx")
y = df["DILI_label"].values
names = list(df["CompoundName"])
smiles = list(df["SMILES"])
mols = [Chem.MolFromSmiles(str(s)) if pd.notna(s) else None for s in smiles]
bitinfos = []
for m in mols:
    bi = {}
    if m is not None:
        AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, bitInfo=bi)
    bitinfos.append(bi)
log(f"[0] 556 化合物，DILI+ = {int(y.sum())}")

# =============================================================================
# (1) 全指纹空间富集（保留指纹位身份）
# =============================================================================
log("\n[1] 全指纹空间富集（fragment := Morgan bit，保留位点身份，count>=5）")
rows = []
for i, (nm, m) in enumerate(zip(names, mols)):
    if m is None:
        continue
    for bit, locations in bitinfos[i].items():
        seen = set()
        for atom_id, radius in locations:
            env = Chem.FindAtomEnvironmentOfRadiusN(m, radius, atom_id)
            if not len(env):
                continue
            sub = Chem.MolToSmiles(Chem.PathToSubmol(m, env))
            if sub in seen:
                continue
            seen.add(sub)
            rows.append({"CompoundName": nm, "DILI_label": int(y[i]),
                         "Fingerprint": f"FP_{bit}", "Substructure_SMILES": sub})
bitmap = pd.DataFrame(rows).drop_duplicates(subset=["CompoundName", "Fingerprint", "Substructure_SMILES"])
log(f"    映射条目 {bitmap.shape}；唯一 (bit, sub) 对 {bitmap.groupby(['Fingerprint','Substructure_SMILES']).ngroups}")

res = []
for (fp, sub), g in bitmap.groupby(["Fingerprint", "Substructure_SMILES"]):
    carriers = set(g["CompoundName"])
    cnt = len(carriers)
    if cnt < MIN_COUNT:
        continue
    pos = int((g["DILI_label"] == 1).sum())
    neg = int((g["DILI_label"] == 0).sum())
    out = ~pd.Series(names).isin(carriers).values
    opos, oneg = int(y[out].sum()), int((y[out] == 0).sum())
    o, lo, hi = or_ci(pos, neg, opos, oneg)
    _, p = fisher_exact([[pos, neg], [opos, oneg]])
    res.append({"Fingerprint": fp, "Substructure_SMILES": sub, "Count": cnt,
                "DILI_positive": pos, "DILI_negative": neg, "DILI_rate": pos / cnt,
                "OddsRatio": o, "OR_CI_low": lo, "OR_CI_high": hi, "p_value": p})
fam = pd.DataFrame(res)
rej, fdr, _, _ = multipletests(fam["p_value"], method="fdr_bh")
fam["FDR"] = fdr
fam["Significant"] = rej
fam = fam.sort_values(["FDR", "OddsRatio"], ascending=[True, False])
fam.to_excel(f"{DATA}/enrichment_fullfamily_bitidentity.xlsx", index=False)
log(f"    检验族大小 n = {len(fam)}")
log(f"    BH FDR 显著数 = {int(fam['Significant'].sum())}；其中 OR>1 = "
    f"{int((fam['Significant'] & (fam['OddsRatio']>1)).sum())}，OR<1 = "
    f"{int((fam['Significant'] & (fam['OddsRatio']<1)).sum())}")
cols = ["Fingerprint", "Substructure_SMILES", "Count", "DILI_rate", "OddsRatio",
        "OR_CI_low", "OR_CI_high", "p_value", "FDR", "Significant"]
hit = fam[fam["Fingerprint"] == "FP_367"]
log("\n    >>> FP_367 在该族中的结果：")
log("    " + hit[cols].to_string(index=False).replace("\n", "\n    "))
if len(hit):
    rk = int((fam["FDR"] <= float(hit["FDR"].iloc[0])).sum())
    log(f"    其 FDR 在族中排名 = {rk} / {len(fam)}")
log("\n    该族 FDR 最显著的前 10 个正富集:")
log("    " + fam[fam["OddsRatio"] > 1][cols].head(10).to_string(index=False).replace("\n", "\n    "))

# =============================================================================
# (2) 直接 SMARTS 检验（不依赖指纹哈希）
# =============================================================================
log("\n[2] 直接 SMARTS 检验（compound-level，556 化合物）")
smarts_defs = {
    "alkyl-substituted aromatic ring (cc(c)C)": "[c][CX4]",
    "aromatic C with methyl branch": "[c][CH3]",
    "aromatic C with any alkyl": "[c][CX4;!$(C=O)]",
    "phenol-like O on aromatic": "[c][OX2H]",
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "primary amide": "[CX3](=O)[NX3H2]",
    "nitro": "[NX3](=O)=O",
    "aniline N on aromatic": "[c][NX3]",
    "chloro on aromatic": "[c][Cl]",
    "aromatic ring only (benzene)": "c1ccccc1",
}
smarts_rows = []
for lab, smt in smarts_defs.items():
    patt = Chem.MolFromSmarts(smt)
    if patt is None:
        log(f"    SMARTS 无效: {lab}")
        continue
    mask = np.array([1 if (m is not None and m.HasSubstructMatch(patt)) else 0 for m in mols])
    smarts_rows.append(test_carriers(mask.astype(bool), y, lab, {"SMARTS": smt}))
sd = pd.DataFrame(smarts_rows)
_, fdr2, _, _ = multipletests(sd["p_value"], method="fdr_bh")
sd["FDR"] = fdr2
sd["Significant"] = fdr2 < 0.05
sd = sd.sort_values("OddsRatio", ascending=False)
sd.to_excel(f"{DATA}/smarts_direct_test.xlsx", index=False)
log(sd[["Definition", "SMARTS", "Count", "DILI_rate", "OddsRatio", "OR_CI_low",
         "OR_CI_high", "p_value", "FDR", "Significant"]].to_string(index=False))

# =============================================================================
# (3) 环境特异性对照：FP_367 携带者 vs 全体 SMARTS 携带者
# =============================================================================
log("\n[3] 环境特异性对照（FP_367 位点 vs 同类裸 SMARTS）")
patt = Chem.MolFromSmarts("[c][CX4]")
mask_smarts = np.array([1 if (m is not None and m.HasSubstructMatch(patt)) else 0 for m in mols]).astype(bool)
mask_bit = np.array([367 in bi for bi in bitinfos])
rows3 = [test_carriers(mask_bit, y, "Morgan bit FP_367 environment"),
         test_carriers(mask_smarts, y, "bare SMARTS [c][CX4] (any alkyl-aryl)"),
         ]
inter = mask_bit & mask_smarts
rows3.append(test_carriers(inter, y, "intersection (bit367 AND SMARTS)"))
only_smarts = mask_smarts & ~mask_bit
rows3.append(test_carriers(only_smarts, y, "SMARTS carriers WITHOUT bit367"))
comp = pd.DataFrame(rows3)
comp.to_excel(f"{DATA}/env_specificity_check.xlsx", index=False)
log(comp.to_string(index=False))
log("\n    解读：若「裸 SMARTS」OR 远低于「FP_367 环境」，说明富集来自具体化学环境而非泛化骨架。")

json.dump({"full_family_bitidentity_size": int(len(fam)),
           "full_family_sig": int(fam["Significant"].sum()),
           "fp367_row": hit.to_dict("records")},
          open(f"{DATA}/family_smarts_summary.json", "w"), indent=2, ensure_ascii=False, default=str)
open(f"{DATA}/22_family_log.txt", "w", encoding="utf-8").write("\n".join(LOG))
log("\n[4] 完成")
