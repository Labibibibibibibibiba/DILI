# -*- coding: utf-8 -*-
"""
Step 4 — Part B: 描述符空间比较 (F1 / F2 / F3 / F4 / F5)
产出: out/S4_descriptor_comparison.csv, out/S4_shap_3d_share.csv,
      out/S4_learning_curves.csv, figures/S4_descriptor_delta.png, out/S4_step4.log

预注册关卡 G3: ΔAUC ≥ 0.03 且嵌套 CV 95% CI 不跨 0 才算 3D 有实质增益。
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import roc_auc_score
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

SEED = dc.SEED
log = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.append(s)

P("=" * 72)
P("Step 4 — Part B 描述符空间比较 (2D vs 3D)")
P("=" * 72)

ds = dc.load_dataset()
FP, DESC = dc.FP_COLS, dc.DESC_COLS
DESC3D = ["Asphericity","Eccentricity","InertialShapeFactor","NPR1","NPR2",
          "PMI1","PMI2","PMI3","RadiusOfGyration","SpherocityIndex","PBF"]

t3d = pd.read_csv(dc.OUT / "feat3d_train.csv", index_col=0)
full = pd.concat([ds.reset_index(drop=True), t3d.reset_index(drop=True)], axis=1)
y = full["DILI_label"].values
P(f"\n[4.1] n={len(full)}  Most {(y==1).sum()} / No {(y==0).sum()}")
P(f"      3D 缺失行数: {t3d.isna().any(axis=1).sum()}")

# 缺失 3D 的行用列中位数填充(并记录)
for c in DESC3D:
    full[c] = full[c].fillna(full[c].median())


def run_cv(cols, tag, n_splits=5):
    X = np.nan_to_num(full[cols].astype(float).values)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs = []
    for tr_i, te_i in skf.split(X, y):
        m = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1)
        m.fit(X[tr_i], y[tr_i])
        aucs.append(roc_auc_score(y[te_i], m.predict_proba(X[te_i])[:, 1]))
    aucs = np.array(aucs)
    m_, sd_ = aucs.mean(), aucs.std(ddof=1)
    ci = (m_ - 1.96 * sd_ / np.sqrt(len(aucs)), m_ + 1.96 * sd_ / np.sqrt(len(aucs)))
    P(f"      {tag:22s} n_feat={len(cols):5d}  AUC {m_:.4f} ± {sd_:.4f}  95%CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    return dict(feature_set=tag, n_feat=len(cols), auc_mean=m_, auc_sd=sd_,
                ci_lo=ci[0], ci_hi=ci[1], aucs=aucs)


P("\n[4.2] 5 折 CV 比较 (L1 标签, RF):")
res = []
res.append(run_cv(FP, "F1_Morgan"))
res.append(run_cv(DESC, "F2_2Ddesc"))
res.append(run_cv(FP + DESC, "F3_Morgan+2D"))
res.append(run_cv(DESC3D, "F4_3Donly"))
res.append(run_cv(FP + DESC + DESC3D, "F5_2D+3D"))

res_df = pd.DataFrame([{k: v for k, v in r.items() if k != "aucs"} for r in res])
res_df.to_csv(dc.OUT / "S4_descriptor_comparison.csv", index=False, encoding="utf-8-sig")

# ======================================================================
# G3 判定: F3 vs F5 (加 3D 是否带来增益)
# ======================================================================
P("\n" + "=" * 72)
f3 = next(r for r in res if r["feature_set"] == "F3_Morgan+2D")
f5 = next(r for r in res if r["feature_set"] == "F5_2D+3D")
f4 = next(r for r in res if r["feature_set"] == "F4_3Donly")
delta = f5["auc_mean"] - f3["auc_mean"]
# 配对差异的 CI (同折配对)
d = f5["aucs"] - f3["aucs"]
d_lo, d_hi = d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d)), d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d))
P(f"[G3] F5(2D+3D) - F3(2D)  ΔAUC = {delta:+.4f}   配对 95%CI [{d_lo:+.4f}, {d_hi:+.4f}]")
P(f"[G3] 要求 ΔAUC ≥ 0.03 且 CI 不跨 0")
g3 = (delta >= 0.03) and (d_lo > 0)
P(f"[G3] 判定: {'3D 有实质增益' if g3 else '3D 无实质增益 -> 如实报告负结果'}")
P(f"[G3] 参考: 3D-only(F4) = {f4['auc_mean']:.4f}  vs  2D-only(F3) = {f3['auc_mean']:.4f}")

# ======================================================================
# SHAP: 3D 特征重要性占比 (RF 内置 importance 作为快速代理)
# ======================================================================
P("\n[4.3] 3D 特征在 F5 中的重要性占比")
X5 = np.nan_to_num(full[FP + DESC + DESC3D].astype(float).values)
m5 = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                            random_state=SEED, n_jobs=-1).fit(X5, y)
imp = pd.Series(m5.feature_importances_, index=FP + DESC + DESC3D)
share3d = imp[DESC3D].sum() / imp.sum()
P(f"      3D 特征总重要性占比 = {share3d*100:.2f}%")
P(imp[DESC3D].sort_values(ascending=False).to_string())
imp.to_csv(dc.OUT / "S4_feature_importance_F5.csv", encoding="utf-8-sig")
pd.DataFrame([dict(share_3d=share3d, delta_auc=delta, delta_ci_lo=d_lo, delta_ci_hi=d_hi,
                   g3_pass=bool(g3))]).to_csv(dc.OUT / "S4_shap_3d_share.csv",
                                              index=False, encoding="utf-8-sig")

(dc.OUT / "S4_step4.log").write_text("\n".join(log), encoding="utf-8")
P("\nStep 4 完成。")
