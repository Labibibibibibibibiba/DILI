#!/usr/bin/env python3
"""
23_external_validation.py
=========================
外部验证（审稿 M4）：在**独立数据集** DILIst（FDA, 1279 药；768 阳性 / 511 阴性）
中检验本文的结构富集结论是否可外推。

关键设计：
  * 只使用 **与训练集无重叠** 的药物（外部子集），确保是真正的独立检验；
  * 结构由 PubChem 按药名解析（带本地缓存，可重复运行）；
  * 检验对象：Morgan bit FP_367 环境、以及若干 SMARTS（含阳性对照与阴性对照）；
  * 报告 OR / 95%CI / Fisher p / BH-FDR。

输出：D:/DILI/data/external/dilist_external_validation.xlsx
      D:/DILI/data/external/pubchem_cache.csv
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

RDLogger.DisableLog("rdApp.*")
EXT = "D:/DILI/data/external"
CACHE = f"{EXT}/pubchem_cache.csv"
LOG = []


def log(m=""):
    print(m, flush=True)
    LOG.append(str(m))


SALT = (r"\s+(sodium|potassium|hydrochloride|hcl|hydrobromide|sulfate|sulphate|mesylate|maleate|"
        r"tartrate|citrate|phosphate|acetate|dihydrate|monohydrate|hydrate|calcium|magnesium|"
        r"besylate|tosylate|fumarate|succinate|bitartrate|gluconate|bromide|chloride|iodide|"
        r"nitrate|oxide|salt|salts|dinitrate|hydrobromide|hyclate|pamoate|embonate)$")


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(SALT, "", s)
    return re.sub(r"\s+", " ", s).strip()


def pubchem_smiles(name, tries=3):
    q = urllib.parse.quote(str(name))
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}"
           f"/property/CanonicalSMILES/TXT")
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=15) as r:
                txt = r.read().decode("utf-8", "ignore").strip().splitlines()
                return txt[0].strip() if txt else None
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                return None
            if e.code == 503:          # 限流：退避
                time.sleep(3.0)
                continue
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
    return None


# =============================================================================
# 1. 外部子集
# =============================================================================
dil = pd.read_excel(f"{EXT}/DILIst.xlsx")
dil.columns = [str(c).strip() for c in dil.columns]
dil = dil.rename(columns={"DILIst Classification": "label"})
dil["label"] = dil["label"].astype(int)
our = pd.read_excel("D:/DILI/data/DILI_ML_dataset_v5_final.xlsx")
our_keys = set(our["CompoundName"].map(norm))
dil["k"] = dil["CompoundName"].map(norm)
ext = dil[~dil["k"].isin(our_keys)].drop_duplicates(subset=["k"]).reset_index(drop=True)
log(f"[1] DILIst {len(dil)} 药 → 外部子集 (与训练集无重叠) {len(ext)} 药；"
    f"阳性 {int(ext['label'].sum())} / 阴性 {int((ext['label']==0).sum())}")

# =============================================================================
# 2. PubChem 解析 SMILES（带缓存）
# =============================================================================
cache = pd.read_csv(CACHE) if os.path.exists(CACHE) else pd.DataFrame(columns=["k", "smiles"])
known = dict(zip(cache["k"], cache["smiles"]))
todo = [k for k in ext["k"] if k not in known]
log(f"[2] PubChem 查询：缓存 {len(known)}，待查 {len(todo)}")
new = []
for i, k in enumerate(todo):
    s = pubchem_smiles(k)
    new.append({"k": k, "smiles": s})
    if (i + 1) % 25 == 0:
        cache = pd.concat([cache, pd.DataFrame(new)], ignore_index=True).drop_duplicates(subset=["k"])
        cache.to_csv(CACHE, index=False)
        log(f"    {i+1}/{len(todo)} 已解析 {int(cache['smiles'].notna().sum())}")
    time.sleep(0.5)
cache = pd.concat([cache, pd.DataFrame(new)], ignore_index=True).drop_duplicates(subset=["k"])
cache.to_csv(CACHE, index=False)
smiles_map = dict(zip(cache["k"], cache["smiles"]))
ext["smiles"] = ext["k"].map(smiles_map)
resolved = ext.dropna(subset=["smiles"])
log(f"    成功解析 {len(resolved)} / {len(ext)}；其中阳性 {int(resolved['label'].sum())} / "
    f"阴性 {int((resolved['label']==0).sum())}")

mols = [Chem.MolFromSmiles(str(s)) for s in resolved["smiles"]]
ok = np.array([m is not None for m in mols])
log(f"    RDKit 可解析 {int(ok.sum())} / {len(mols)}")
y = resolved["label"].values[ok]
mols = [m for m, o in zip(mols, ok) if o]
names = resolved["k"].values[ok]

bitinfos = []
for m in mols:
    bi = {}
    AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, bitInfo=bi)
    bitinfos.append(bi)
log(f"    最终外部检验集 n = {len(mols)}（阳性 {int(y.sum())} / 阴性 {int((y==0).sum())}）")


def or_ci(a, b, c, d):
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    o = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(o), float(np.exp(np.log(o) - 1.96 * se)), float(np.exp(np.log(o) + 1.96 * se))


def test(mask, label, extra=None):
    mask = np.asarray(mask).astype(bool)
    a, b = int((mask & (y == 1)).sum()), int((mask & (y == 0)).sum())
    c, d = int((~mask & (y == 1)).sum()), int((~mask & (y == 0)).sum())
    o, lo, hi = or_ci(a, b, c, d)
    _, p = fisher_exact([[a, b], [c, d]])
    row = {"Definition": label, "Count": a + b, "DILI_positive": a, "DILI_negative": b,
           "DILI_rate": a / (a + b) if (a + b) else np.nan, "OddsRatio": o,
           "OR_CI_low": lo, "OR_CI_high": hi, "p_value": p}
    if extra:
        row.update(extra)
    return row


rows = []
# (a) 指纹位环境（含阳性对照 FP_367 与若干阴性对照）
for bit, lab in [(367, "Morgan bit FP_367 (cc(c)C env)"),
                 (981, "Morgan bit FP_981 (CN env)"),
                 (356, "Morgan bit FP_356"),
                 (849, "Morgan bit FP_849")]:
    rows.append(test([bit in bi for bi in bitinfos], lab, {"Type": "fingerprint_bit"}))

# (b) SMARTS
smarts_defs = {
    "[c][CH3]": "aromatic C with methyl branch",
    "[c][NX3]": "aniline N on aromatic ring",
    "[c][Cl]": "aryl chloride",
    "[c][CX4]": "alkyl-substituted aromatic ring (generic)",
    "c1ccccc1": "benzene ring (generic control)",
    "[CX3](=O)[OX2H1]": "carboxylic acid (control)",
}
for smt, lab in smarts_defs.items():
    patt = Chem.MolFromSmarts(smt)
    m_ = [1 if (m is not None and m.HasSubstructMatch(patt)) else 0 for m in mols]
    rows.append(test(m_, lab, {"Type": "smarts", "SMARTS": smt}))

r = pd.DataFrame(rows)
rej, fdr, _, _ = multipletests(r["p_value"], method="fdr_bh")
r["FDR"] = fdr
r["Significant"] = rej
r = r.sort_values("OddsRatio", ascending=False)
r.to_excel(f"{EXT}/dilist_external_validation.xlsx", index=False)
log("\n[3] 外部验证结果（DILIst 独立子集）:")
log(r[["Definition", "Type", "Count", "DILI_rate", "OddsRatio", "OR_CI_low",
       "OR_CI_high", "p_value", "FDR", "Significant"]].to_string(index=False))

# =============================================================================
# 3. 与训练集内部结果并列
# =============================================================================
inner = {"FP_367 env (train 556)": {"OR": 4.893, "CI": [2.126, 11.265], "p": 9.1e-5, "FDR": 0.0067},
         "[c][CH3] (train 556)": {"OR": 3.001, "CI": [1.738, 5.182], "p": 6.9e-5, "FDR": 3.5e-4},
         "[c][NX3] (train 556)": {"OR": 2.922, "CI": [2.003, 4.263], "p": 2.7e-8, "FDR": 2.7e-7},
         "[c][CX4] (train 556)": {"OR": 1.139, "CI": [0.803, 1.615], "p": 4.8e-1, "FDR": 5.3e-1}}
log("\n[4] 内部 vs 外部并列表:")
log(f"    {'definition':38s} {'inner OR':>9s} {'extern OR':>10s} {'extern 95%CI':>20s} {'extern p':>10s} {'exterFDR':>9s}")
for lab, iv in inner.items():
    ext_key = lab.split(" (train")[0]
    hit = r[r["Definition"].str.startswith(ext_key.replace("[c]", "[c]"))]
    if len(hit):
        h = hit.iloc[0]
        log(f"    {lab:38s} {iv['OR']:9.3f} {h['OddsRatio']:10.3f} "
            f"{'[%.2f, %.2f]'%(h['OR_CI_low'],h['OR_CI_high']):>20s} {h['p_value']:10.2e} {h['FDR']:9.4f}")
    else:
        log(f"    {lab:38s} {iv['OR']:9.3f}   (外部未匹配)")

json.dump({"n_external": int(len(mols)), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
           "results": r.to_dict("records")},
          open(f"{EXT}/external_validation_summary.json", "w"), indent=2, ensure_ascii=False, default=str)
open(f"{EXT}/23_external_log.txt", "w", encoding="utf-8").write("\n".join(LOG))
log("\n[5] 完成")
