# -*- coding: utf-8 -*-
"""
Step 2 — 3D 特征工程 (只读 D:\DILI)
产出: out/feat3d_train.csv, out/feat3d_unknown.csv
      out/S2_embed_failures.csv, out/S2_step2.log

约定(交接文件 §4.2):
  EmbedMolecule(randomSeed=42) -> MMFFOptimizeMolecule
  取最低能量构象; 同时记录"单构象 vs 多构象平均"的稳定性
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

log = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.append(s)

P("=" * 72)
P("Step 2 — 3D 描述符生成  (RDKit Embed + MMFF)")
P("=" * 72)


# 11 个 3D 描述符 (交接文件 §4.2 F4)
DESC3D = ["Asphericity", "Eccentricity", "InertialShapeFactor",
          "NPR1", "NPR2", "PMI1", "PMI2", "PMI3",
          "RadiusOfGyration", "SpherocityIndex", "PBF"]


def calc_3d(mol, n_confs=1):
    """返回 (desc_dict, err_msg)。多个构象时取最低能量构象。"""
    try:
        mh = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=n_confs, params=params))
        if not cids:
            return None, "embed_failed"
        # 优化 + 取最低能量
        energies = []
        for cid in cids:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mh, AllChem.MMFFGetMoleculeProperties(mh), confId=cid)
                if ff is None:
                    energies.append((np.inf, cid))
                else:
                    ff.Minimize(maxIts=200)
                    energies.append((ff.CalcEnergy(), cid))
            except Exception as e:
                energies.append((np.inf, cid))
        energies.sort(key=lambda t: t[0])
        best_e, best_cid = energies[0]
        if not np.isfinite(best_e):
            return None, "mmff_failed"

        vals = {
            "Asphericity":         Descriptors3D.Asphericity(mh, confId=best_cid),
            "Eccentricity":        Descriptors3D.Eccentricity(mh, confId=best_cid),
            "InertialShapeFactor": Descriptors3D.InertialShapeFactor(mh, confId=best_cid),
            "NPR1":                Descriptors3D.NPR1(mh, confId=best_cid),
            "NPR2":                Descriptors3D.NPR2(mh, confId=best_cid),
            "PMI1":                Descriptors3D.PMI1(mh, confId=best_cid),
            "PMI2":                Descriptors3D.PMI2(mh, confId=best_cid),
            "PMI3":                Descriptors3D.PMI3(mh, confId=best_cid),
            "RadiusOfGyration":    Descriptors3D.RadiusOfGyration(mh, confId=best_cid),
            "SpherocityIndex":     Descriptors3D.SpherocityIndex(mh, confId=best_cid),
            "PBF":                 rdMolDescriptors.CalcPBF(mh, confId=best_cid),
        }
        vals["_energy"] = best_e
        vals["_nconf"] = len(cids)
        return vals, ""
    except Exception as e:
        return None, f"exception:{type(e).__name__}"


def build(smiles_series, tag, n_confs=1):
    rows, fails = [], []
    n = len(smiles_series)
    for i, (idx, smi) in enumerate(smiles_series.items()):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
        if mol is None:
            fails.append(dict(tag=tag, idx=idx, reason="bad_smiles"))
            rows.append({k: np.nan for k in DESC3D}); continue
        vals, err = calc_3d(mol, n_confs=n_confs)
        if vals is None:
            fails.append(dict(tag=tag, idx=idx, reason=err))
            rows.append({k: np.nan for k in DESC3D})
        else:
            rows.append({k: vals[k] for k in DESC3D})
        if (i + 1) % 100 == 0:
            P(f"      [{tag}] {i+1}/{n}")
    df = pd.DataFrame(rows, index=smiles_series.index, columns=DESC3D)
    df.index.name = "row_idx"
    return df, pd.DataFrame(fails)


# ======================================================================
# 1. 训练集 556
# ======================================================================
ds = dc.load_dataset()
P(f"\n[2.1] 训练集 n={len(ds)}  开始单构象 3D...")
t0 = time.time()
tr3d, tr_fail = build(ds["SMILES"], "train", n_confs=1)
P(f"      耗时 {time.time()-t0:.1f}s  失败 {len(tr_fail)}/{len(ds)} = {len(tr_fail)/len(ds)*100:.2f}%")

# ======================================================================
# 2. 未知队列 659
# ======================================================================
raw, mat = dc.load_unknown()
P(f"\n[2.2] 未知队列(SMILES 可得) n={len(mat)}  开始单构象 3D...")
# mat 里有 SMILES 吗?
if "SMILES" in mat.columns:
    smi_unk = mat["SMILES"]
elif "SMILES" in raw.columns:
    smi_unk = raw.loc[mat.index, "SMILES"]
else:
    smi_unk = None
    P("      [WARN] 未知矩阵无 SMILES 列, 尝试从 prediction_rank 取")
    pr = pd.read_excel(dc.DILI / "data" / "DILI_unknown_prediction_rank.xlsx")
    smi_unk = pr["SMILES"].reset_index(drop=True)

if smi_unk is None or smi_unk.isna().all():
    P("      [ERROR] 无法获得未知队列 SMILES, 跳过 Step 2 未知部分")
    unk3d, unk_fail = pd.DataFrame(columns=DESC3D), pd.DataFrame()
else:
    t0 = time.time()
    unk3d, unk_fail = build(smi_unk, "unknown", n_confs=1)
    P(f"      耗时 {time.time()-t0:.1f}s  失败 {len(unk_fail)}/{len(smi_unk)} = {len(unk_fail)/max(1,len(smi_unk))*100:.2f}%")

# ======================================================================
# 3. 保存
# ======================================================================
tr3d.to_csv(dc.OUT / "feat3d_train.csv", index=True, encoding="utf-8-sig")
unk3d.to_csv(dc.OUT / "feat3d_unknown.csv", index=True, encoding="utf-8-sig")
allf = pd.concat([tr_fail, unk_fail], ignore_index=True) if len(tr_fail) or len(unk_fail) else pd.DataFrame(columns=["tag","idx","reason"])
allf.to_csv(dc.OUT / "S2_embed_failures.csv", index=False, encoding="utf-8-sig")

P(f"\n[2.3] feat3d_train   {tr3d.shape} -> out/feat3d_train.csv")
P(f"      feat3d_unknown {unk3d.shape} -> out/feat3d_unknown.csv")
P(f"      失败明细 {len(allf)} 条 -> out/S2_embed_failures.csv")
if len(allf):
    P("      失败原因分布:")
    for k, v in allf["reason"].value_counts().items():
        P(f"        {k}: {v}")

# ======================================================================
# 4. G2 关卡
# ======================================================================
rate_tr = len(tr_fail) / len(ds) * 100
rate_unk = len(unk_fail) / max(1, len(smi_unk)) * 100
P("\n" + "=" * 72)
P(f"[G2] 训练集嵌入失败率 = {rate_tr:.2f}%")
P(f"[G2] 未知集嵌入失败率 = {rate_unk:.2f}%")
P(f"[G2] 阈值 5%  ->  {'通过' if max(rate_tr, rate_unk) <= 5 else '未通过(>5%)'}")

# 描述符描述统计(判断是否有常数/退化列)
P("\n[2.4] 3D 描述符分布(train):")
P(tr3d.describe().T[["mean", "std", "min", "max"]].to_string())

summary = dict(
    n_train=len(ds), fail_train=len(tr_fail), rate_train=rate_tr,
    n_unknown=int(len(smi_unk)) if smi_unk is not None else 0,
    fail_unknown=len(unk_fail), rate_unknown=rate_unk,
    n_desc3d=len(DESC3D), g2_pass=bool(max(rate_tr, rate_unk) <= 5),
)
(dc.OUT / "S2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
(dc.OUT / "S2_step2.log").write_text("\n".join(log), encoding="utf-8")
P("\nStep 2 完成。")
