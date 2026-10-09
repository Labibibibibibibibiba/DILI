# -*- coding: utf-8 -*-
"""量化 F1(纯 Morgan) vs F3(Morgan+7描述符) 的差异在未知集排名上的实际后果。
只读 D:\DILI, 只写 D:\DILI_1。口径: 训练池 = 556 全量, L1 编码, RF。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr, mannwhitneyu

out = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.append(s)

ds = dc.load_dataset()
raw, mat = dc.load_unknown()
FP, DESC = dc.FP_COLS, dc.DESC_COLS

unk = mat.copy()
pr = pd.read_excel(dc.DILI / "data" / "DILI_unknown_prediction_rank.xlsx")
pr["_k"] = pr["CompoundName"].astype(str).str.strip()
unk["_k"] = unk["CompoundName"].astype(str).str.strip()
unk["cat"] = unk["_k"].map(dict(zip(pr["_k"], pr["DILI_category"]))).astype(str).str.strip().map(
    {"vLess-DILI-concern": "vLess", "Ambiguous-DILI-concern": "Ambiguous"})
assert unk["cat"].notna().all()

Xtr = ds[FP + DESC].astype(float).values
y = ds["DILI_label"].values
names = unk["CompoundName"].values


def score(cols):
    Xc = np.nan_to_num(unk[cols].astype(float).values)
    m = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                               random_state=dc.SEED, n_jobs=-1).fit(
        np.nan_to_num(ds[cols].astype(float).values), y)
    return m.predict_proba(Xc)[:, 1], m


s1, m1 = score(FP)
s3, m3 = score(FP + DESC)
r1 = pd.Series(s1).rank(ascending=False, method="min").values
r3 = pd.Series(s3).rank(ascending=False, method="min").values

rho, pv = spearmanr(s1, s3)
P(f"[A] 659 未知集: F1 vs F3 分数 Spearman rho = {rho:.4f}  (P={pv:.2e})")
P(f"    排名位移 |dr|: 中位={np.median(np.abs(r1-r3)):.1f}  "
  f"90分位={np.percentile(np.abs(r1-r3),90):.1f}  最大={np.abs(r1-r3).max():.0f}")

# Top-k 集合一致性
for k in [16, 33, 66, 132]:
    a, b = set(np.argsort(-s1)[:k]), set(np.argsort(-s3)[:k])
    P(f"    Top-{k:3d} 重叠 {len(a & b)}/{k}  (Jaccard {len(a&b)/len(a|b):.3f})")

# 尾部富集是否都成立
for tag, s in [("F1", s1), ("F3", s3)]:
    top = np.argsort(-s)[:33]
    vl = (unk["cat"].values[top] == "vLess").sum()
    P(f"    {tag} Top5%: vLess {vl}/33")

# 候选药分数
P("\n[B] V2.2 核心 7 药 (L1 编码 / 全量训练池 / RF)")
targets = ["Chlorpromazine hydrochloride", "Sulfamethizole", "Cabotegravir sodium",
           "Mebendazole", "Oxybutynin chloride", "Tolterodine tartrate",
           "Metformin hydrochloride"]
idx = {n: i for i, n in enumerate(names)}
P(f"    {'药':32s}{'F1分':>8s}{'F3分':>8s}{'F1排':>6s}{'F3排':>6s}{'Δ排':>7s}")
for t in targets:
    if t not in idx:
        P(f"    {t:32s} 未找到"); continue
    i = idx[t]
    P(f"    {t:32s}{s1[i]:>8.3f}{s3[i]:>8.3f}{r1[i]:>6.0f}{r3[i]:>6.0f}{r1[i]-r3[i]:>+7.0f}")

# 特征重要性: 7 描述符块占比
imp = pd.Series(m3.feature_importances_, index=FP + DESC)
P(f"\n[C] F3 空间 RF 块重要性: bit={imp[FP].sum():.4f}  7desc={imp[DESC].sum():.4f}  "
  f"→ 1024 bit 占 {100*imp[FP].sum()/imp.sum():.1f}%, 7 desc 占 {100*imp[DESC].sum()/imp.sum():.1f}%")
P(f"    每维平均重要性: bit {imp[FP].mean():.6f} vs desc {imp[DESC].mean():.6f}  "
  f"(描述符单维是 bit 的 {imp[DESC].mean()/imp[FP].mean():.0f} 倍)")

# 分数分位分布: 描述符改变了分布吗
P("\n[D] 分数分布 (659)")
P(f"    F1: min={s1.min():.3f} 中位={np.median(s1):.3f} max={s1.max():.3f} sd={s1.std():.3f}")
P(f"    F3: min={s3.min():.3f} 中位={np.median(s3):.3f} max={s3.max():.3f} sd={s3.std():.3f}")

(Path(dc.OUT) / "_q_F1vsF3.log").write_text("\n".join(out), encoding="utf-8")
