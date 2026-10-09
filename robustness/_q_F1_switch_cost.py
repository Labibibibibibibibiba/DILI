# -*- coding: utf-8 -*-
"""回答: 若改主模型为纯 Morgan(F1), 论文各输出会变化多少?
1) 659 未知集具体名次位移
2) Top5% 尾部富集是否复现
3) DILIst 外部验证 FP_367 富集是否复现
只读 D:\DILI。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import fisher_exact, mannwhitneyu
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

out = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.append(s)

ds = dc.load_dataset()
raw, mat = dc.load_unknown()
FP, DESC = dc.FP_COLS, dc.DESC_COLS
y = ds["DILI_label"].values

pr = pd.read_excel(dc.DILI / "data" / "DILI_unknown_prediction_rank.xlsx")
pr["_k"] = pr["CompoundName"].astype(str).str.strip()
mat["_k"] = mat["CompoundName"].astype(str).str.strip()
mat["cat"] = mat["_k"].map(dict(zip(pr["_k"], pr["DILI_category"]))).astype(str).str.strip().map(
    {"vLess-DILI-concern": "vLess", "Ambiguous-DILI-concern": "Ambiguous"})
assert mat["cat"].notna().all(), "类别映射失败"

names = mat["CompoundName"].values
cats = mat["cat"].values


def fit_score(cols):
    m = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                               random_state=dc.SEED, n_jobs=-1)
    m.fit(np.nan_to_num(ds[cols].astype(float).values), y)
    s = m.predict_proba(np.nan_to_num(mat[cols].astype(float).values))[:, 1]
    return s, m

s1, _ = fit_score(FP)
s3, m3 = fit_score(FP + DESC)
r1 = pd.Series(s1).rank(ascending=False, method="min").values.astype(int)
r3 = pd.Series(s3).rank(ascending=False, method="min").values.astype(int)

P("=" * 70)
P("[1] Top5% 尾部富集 (论文 §3.5 核心应用结论)")
P("=" * 70)
for tag, s, r in [("F1 纯Morgan", s1, r1), ("F3 主模型", s3, r3)]:
    top = np.argsort(-s)[:33]
    vl = int((cats[top] == "vLess").sum()); am = 33 - vl
    nvl = int((cats == "vLess").sum()); nam = int((cats == "Ambiguous").sum())
    tbl = [[vl, am], [nvl - vl, nam - am]]
    orr, p = fisher_exact(tbl)
    P(f"  {tag:12s} Top5%(33) = vLess {vl} + Amb {am}   OR={orr:.2f}  Fisher P={p:.4f}  "
      f"{'✅显著' if p<0.05 else '❌不显著'}")

P("")
P("=" * 70)
P("[2] 全部类别分布 Mann-Whitney (论文 §3.5)")
P("=" * 70)
for tag, s in [("F1", s1), ("F3", s3)]:
    v, a = s[cats == "vLess"], s[cats == "Ambiguous"]
    u, p = mannwhitneyu(v, a)
    P(f"  {tag}: vLess 中位={np.median(v):.3f}  Amb 中位={np.median(a):.3f}  MWU P={p:.4f}")

P("")
P("=" * 70)
P("[3] V2.2 核心 7 药名次 (论文 §3.5 点名药物)")
P("=" * 70)
tg = {"Chlorpromazine hydrochloride": "CPZ", "Sulfamethizole": "SMX",
      "Cabotegravir sodium": "CAB", "Mebendazole": "MBZ",
      "Oxybutynin chloride": "OXY", "Tolterodine tartrate": "TOL",
      "Metformin hydrochloride": "MET"}
idx = {n: i for i, n in enumerate(names)}
P(f"  {'药':6s}{'F1名次':>8s}{'F3名次':>8s}{'位移':>8s}{'F1分数':>9s}{'F3分数':>9s}")
for full, ab in tg.items():
    if full not in idx:
        P(f"  {ab:6s} 未找到"); continue
    i = idx[full]
    P(f"  {ab:6s}{r1[i]:>8d}{r3[i]:>8d}{r1[i]-r3[i]:>+8d}{s1[i]:>9.3f}{s3[i]:>9.3f}")

# 前 20 名逐名对照
P("")
P("[3b] Top-20 名单直接可比性")
t1 = [names[i] for i in np.argsort(-s1)[:20]]
t3 = [names[i] for i in np.argsort(-s3)[:20]]
same_pos = sum(1 for i in range(20) if t1[i] == t3[i])
P(f"   Top-20 中同名列 {len(set(t1)&set(t3))}/20, 名次完全相同 {same_pos}/20")
for i in range(20):
    mark = "=" if t1[i] == t3[i] else " "
    P(f"   {i+1:2d} {t1[i][:34]:36s} | {t3[i][:34]:36s} {mark}")

P("")
P("=" * 70)
P("[4] FP_367 的 RF 内置重要性 (非 SHAP)")
P("=" * 70)
imp3 = pd.Series(m3.feature_importances_, index=FP + DESC)
rk = imp3[FP].rank(ascending=False)
P(f"  F3 空间 FP_367 重要性 {imp3['FP_367']:.6f}, 在 1024 bit 中排第 {int(rk['FP_367'])} 位")
P(f"  bit 块总重要性 {imp3[FP].sum():.4f} / desc 块 {imp3[DESC].sum():.4f}")
imp1 = pd.Series(fit_score(FP)[1].feature_importances_, index=FP)
P(f"  F1 空间 FP_367 重要性 {imp1['FP_367']:.6f}, 排第 {int(imp1.rank(ascending=False)['FP_367'])} 位")

P("")
P("=" * 70)
P("[5] 外部验证 DILIst: FP_367 富集 (论文 §3.6)")
P("=" * 70)
try:
    ext = pd.read_excel(dc.DILI / "data" / "DILIst_external_validation.xlsx")
    P(f"  读取成功 {ext.shape} 列={list(ext.columns)[:8]}")
except Exception:
    cands = list((dc.DILI / "data").glob("*DILIst*")) + list((dc.DILI / "data").glob("*dilist*"))
    P(f"  未找到标准文件名, 目录中含 DILIst 的文件: {[c.name for c in cands][:10]}")

(Path(dc.OUT) / "_q_F1_switch_cost.log").write_text("\n".join(out), encoding="utf-8")
