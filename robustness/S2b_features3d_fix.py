# -*- coding: utf-8 -*-
"""
Step 2b — 3D 嵌入失败根因诊断与补救尝试 (只读 D:\DILI)
问题: MMFF 优化失败 236/264。尝试:
  方案A: MMFF 失败时回退 UFF
  方案B: 力场全失败时, 直接用 ETKDG 嵌入构象(不优化)
  方案C: 限制构象生成次数 + 加大 maxAttempts
目标: 把失败率压到 ≤5%
"""
import sys, time
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
P("Step 2b — 3D 嵌入补救尝试")
P("=" * 72)

DESC3D = ["Asphericity","Eccentricity","InertialShapeFactor","NPR1","NPR2",
          "PMI1","PMI2","PMI3","RadiusOfGyration","SpherocityIndex","PBF"]


def calc_3d_v2(mol):
    """MMFF -> UFF -> 无优化, 逐级回退。返回 (vals, method)。"""
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    try:
        params.maxIterations = 50     # 新版 RDKit 参数名
    except Exception:
        pass
    cid = AllChem.EmbedMolecule(mh, params)
    if cid < 0:
        # 再试一次, 用随机坐标做种子 + 放宽
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mh, params)
        if cid < 0:
            return None, "embed_failed"

    method = None
    # 1) MMFF
    try:
        props = AllChem.MMFFGetMoleculeProperties(mh)
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mh, props, confId=cid)
            if ff is not None and ff.Minimize(maxIts=500) == 0:
                method = "MMFF"
    except Exception:
        pass
    # 2) UFF 回退
    if method is None:
        try:
            ff = AllChem.UFFGetMoleculeForceField(mh, confId=cid)
            if ff is not None and ff.Minimize(maxIts=500) == 0:
                method = "UFF"
        except Exception:
            pass
    # 3) 无优化(直接用嵌入构象)
    if method is None:
        method = "ETKDG_noopt"

    try:
        vals = {
            "Asphericity":         Descriptors3D.Asphericity(mh, confId=cid),
            "Eccentricity":        Descriptors3D.Eccentricity(mh, confId=cid),
            "InertialShapeFactor": Descriptors3D.InertialShapeFactor(mh, confId=cid),
            "NPR1":                Descriptors3D.NPR1(mh, confId=cid),
            "NPR2":                Descriptors3D.NPR2(mh, confId=cid),
            "PMI1":                Descriptors3D.PMI1(mh, confId=cid),
            "PMI2":                Descriptors3D.PMI2(mh, confId=cid),
            "PMI3":                Descriptors3D.PMI3(mh, confId=cid),
            "RadiusOfGyration":    Descriptors3D.RadiusOfGyration(mh, confId=cid),
            "SpherocityIndex":     Descriptors3D.SpherocityIndex(mh, confId=cid),
            "PBF":                 rdMolDescriptors.CalcPBF(mh, confId=cid),
        }
        return vals, method
    except Exception as e:
        return None, f"desc_failed:{type(e).__name__}"


def build(smi_series, tag):
    rows, fails, methods = [], [], []
    for i, (idx, smi) in enumerate(smi_series.items()):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
        if mol is None:
            fails.append(dict(tag=tag, idx=idx, reason="bad_smiles"))
            rows.append({k: np.nan for k in DESC3D}); methods.append("bad_smiles"); continue
        vals, mth = calc_3d_v2(mol)
        if vals is None:
            fails.append(dict(tag=tag, idx=idx, reason=mth))
            rows.append({k: np.nan for k in DESC3D})
        else:
            rows.append({k: vals[k] for k in DESC3D})
        methods.append(mth)
        if (i + 1) % 100 == 0:
            P(f"      [{tag}] {i+1}/{len(smi_series)}")
    df = pd.DataFrame(rows, index=smi_series.index, columns=DESC3D)
    df.index.name = "row_idx"
    return df, pd.DataFrame(fails), pd.Series(methods, index=smi_series.index)


ds = dc.load_dataset()
raw, mat = dc.load_unknown()

P(f"\n[2b.1] 先在训练集前 200 个上试跑...")
sub = ds["SMILES"].head(200)
t0 = time.time()
_, f_sub, m_sub = build(sub, "probe")
P(f"      耗时 {time.time()-t0:.1f}s  失败 {len(f_sub)}/{len(sub)} = {len(f_sub)/len(sub)*100:.2f}%")
P(f"      方法分布: {m_sub.value_counts().to_dict()}")

P("\n[2b.2] 若失败率 ≤5%, 全量重跑训练集 + 未知集")
if len(f_sub) / len(sub) <= 0.05:
    t0 = time.time()
    tr3d, tr_f, tr_m = build(ds["SMILES"], "train")
    P(f"      train 耗时 {time.time()-t0:.1f}s 失败 {len(tr_f)}/{len(ds)} = {len(tr_f)/len(ds)*100:.2f}%")
    P(f"      train 方法分布: {tr_m.value_counts().to_dict()}")

    # 未知集 SMILES
    pr = pd.read_excel(dc.DILI / "data" / "DILI_unknown_prediction_rank.xlsx")
    smi_unk = pr["SMILES"] if "SMILES" in pr.columns else mat.get("SMILES")
    t0 = time.time()
    unk3d, unk_f, unk_m = build(smi_unk.reset_index(drop=True), "unknown")
    P(f"      unknown 耗时 {time.time()-t0:.1f}s 失败 {len(unk_f)}/{len(smi_unk)} = {len(unk_f)/len(smi_unk)*100:.2f}%")
    P(f"      unknown 方法分布: {unk_m.value_counts().to_dict()}")

    # 保存为 v2 (不覆盖失败的旧版, 但仍写主文件名以便后续步骤使用)
    tr3d.to_csv(dc.OUT / "feat3d_train.csv", index=True, encoding="utf-8-sig")
    unk3d.to_csv(dc.OUT / "feat3d_unknown.csv", index=True, encoding="utf-8-sig")
    pd.concat([tr_f, unk_f], ignore_index=True).to_csv(
        dc.OUT / "S2_embed_failures.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"tag": ["train"]*len(tr_m) + ["unknown"]*len(unk_m),
                  "method": list(tr_m.values) + list(unk_m.values)}).to_csv(
        dc.OUT / "S2_embed_methods.csv", index=False, encoding="utf-8-sig")

    P("\n" + "=" * 72)
    P(f"[G2-v2] train 失败率 {len(tr_f)/len(ds)*100:.2f}%  unknown 失败率 {len(unk_f)/len(smi_unk)*100:.2f}%")
    P(f"[G2-v2] 判定: {'通过' if max(len(tr_f)/len(ds), len(unk_f)/len(smi_unk)) <= 0.05 else '仍未通过'}")
else:
    P(f"      试跑失败率 {len(f_sub)/len(sub)*100:.2f}% > 5%, 补救无效")

pd.DataFrame(f_sub if len(f_sub) else [], columns=["tag","idx","reason"]).to_csv(
    dc.OUT / "S2b_probe_failures.csv", index=False, encoding="utf-8-sig")
(dc.OUT / "S2b_step2b.log").write_text("\n".join(log), encoding="utf-8")
P("\nStep 2b 完成。")
