# -*- coding: utf-8 -*-
"""一次性查询: 主项目数据表特征列结构 + 7 描述符 vs 1024 bit 的量纲/方差/信息冗余量化。
只读 D:\DILI, 不写 D:\DILI 任何文件。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

out = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.append(s)

df = dc.load_dataset()
FP, DESC = dc.FP_COLS, dc.DESC_COLS
P(f"[表结构] 列数={df.shape[1]}  列名前8={list(df.columns[:8])}")
P(f"  SMILES 在表中: {'SMILES' in df.columns}")
missing = [c for c in FP if c not in df.columns]
P(f"  FP 列缺失数: {len(missing)}")
P(f"  DESC 列: {DESC}")

y = df["DILI_label"].values if "DILI_label" in df.columns else None
P(f"  标签: Most {int((y==1).sum())} / No {int((y==0).sum())}  n={len(df)}")

# --- 1. 量纲/方差 ---
P("\n[1] 原始尺度与方差 (未标准化)")
rows = []
for c in DESC:
    v = df[c].astype(float).values
    rows.append((c, v.min(), v.max(), v.mean(), v.std(),
                 (v == v.min()).mean(), (v == v.max()).mean()))
for r in rows:
    P(f"  {r[0]:18s} min={r[1]:9.2f} max={r[2]:10.2f} mean={r[3]:8.2f} sd={r[4]:8.3f}")
fp = df[FP].astype(float).values
P(f"  {'FP_all':18s} min={fp.min()} max={fp.max()} mean={fp.mean():.4f} sd={fp.mean(axis=0).std():.4f}")
P(f"  FP 每 bit 出现率的分布: min={fp.mean(axis=0).min():.4f} "
  f"中位={np.median(fp.mean(axis=0)):.4f} max={fp.mean(axis=0).max():.4f}")
P(f"  出现率>0.5 的 bit 数: {(fp.mean(axis=0)>0.5).sum()} / 1024")
P(f"  出现率<0.01 的 bit 数: {(fp.mean(axis=0)<0.01).sum()} / 1024")

# --- 2. LR 是否被量纲压垮 ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=dc.SEED)

def cv_auc(X, tag, model):
    a = []
    for tr, te in cv.split(X, y):
        m = model()
        m.fit(X[tr], y[tr])
        a.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    a = np.array(a)
    P(f"  {tag:34s} AUC {a.mean():.4f} ± {a.std(ddof=1):.4f}")
    return a

P("\n[2] LR: 未标准化 vs 标准化 (检验 '唯一增益是尺度' 假说)")
X_fp = df[FP].astype(float).values
X_fpdesc = df[FP + DESC].astype(float).values
X_desc = df[DESC].astype(float).values
L = lambda: LogisticRegression(max_iter=5000, class_weight="balanced", random_state=dc.SEED)
cv_auc(X_fp,     "LR  Morgan(原始,均0-1)", L)
cv_auc(X_fpdesc, "LR  Morgan+7desc(原始)", L)
cv_auc(X_desc,   "LR  仅7desc(原始)", L)
Z = lambda: make_pipeline(StandardScaler(), L())
cv_auc(X_fp,     "LR  Morgan(标准化)", Z)
cv_auc(X_fpdesc, "LR  Morgan+7desc(标准化)", Z)
cv_auc(X_desc,   "LR  仅7desc(标准化)", Z)

# --- 3. 描述符是否已被 bit 张成 ---
P("\n[3] 描述符能否由 bit 子集线性重建 (R^2, 岭回归)")
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
Xs = StandardScaler().fit_transform(X_fp)
for c in DESC:
    t = df[c].astype(float).values
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 20)), Xs, t, cv=5)
    ss_res = ((t - pred) ** 2).sum(); ss_tot = ((t - t.mean()) ** 2).sum()
    P(f"  {c:18s} R^2 = {1 - ss_res/ss_tot:6.3f}")

# --- 4. 逐位点删除: 加描述符后 RF 的净效应 ---
P("\n[4] 特征块重要性 (RF 内置 importance, F3 空间)")
m3 = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                            random_state=dc.SEED, n_jobs=-1).fit(X_fpdesc, y)
imp = pd.Series(m3.feature_importances_, index=FP + DESC)
P(f"  bit 块总重要性 = {imp[FP].sum():.4f}   7desc 块总重要性 = {imp[DESC].sum():.4f}")
P("  7 描述符各自重要性 (按降序):")
P(imp[DESC].sort_values(ascending=False).to_string())

# --- 5. 描述符与 SHAP 主题是否一致 ---
P("\n[5] 描述符单独的相关性 (打分方向)")
from scipy.stats import mannwhitneyu
for c in DESC:
    v = df[c].astype(float).values
    pos, neg = v[y == 1], v[y == 0]
    u, p = mannwhitneyu(pos, neg)
    P(f"  {c:18s} Most中位={np.median(pos):8.2f} No中位={np.median(neg):8.2f}  P={p:.2e}")

Path(dc.OUT / "_q_feat_space.log").write_text("\n".join(out), encoding="utf-8")
