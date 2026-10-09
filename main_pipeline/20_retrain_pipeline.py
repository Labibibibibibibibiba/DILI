#!/usr/bin/env python3
"""
20_retrain_pipeline.py
======================
重建训练/评估管线（v2），修复审稿 M1「决策阈值在测试集上选取」的泄漏问题。

决策依据（用户 2026-10-03 拍板）：
  1. class_weight 保留 "balanced"（不改为手工加权）；
  2. 走「重算路线」：阈值在训练集 out-of-fold (OOF) 预测上选取，只对测试集评估一次；
  3. 外部验证另见 21_livertox_validation.py。

本脚本产出（全部落盘到 D:/DILI/data/）：
  - best_RF_model_v2.pkl          最终 RF 模型
  - RF_params_v2.json             最终超参数 + 搜索空间
  - model_metrics_v2.json         全部指标（CV / OOF 阈值 / 测试集 / scaffold / bootstrap CI）
  - cv_folds_v2.xlsx              LR/RF/XGB 每折 CV AUC（训练集 CV 与全 556 CV 两套口径）
  - threshold_oof_v2.xlsx         训练集 OOF 阈值扫描表
  - environment_v2.json           包版本 + 随机种子

纪律：所有随机种子显式固定；所有关键数字均在本脚本内重新计算，不引用旧值。
"""

import json
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from xgboost import XGBClassifier

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

np.random.seed(42)

DATA = "D:/DILI/data"
SEEDS = {"split": 42, "cv": 42, "tuning": 42, "scaffold": 42, "bootstrap": 20261003}

t0 = time.time()
log_lines = []


def log(msg=""):
    print(msg, flush=True)
    log_lines.append(str(msg))


# =============================================================================
# 1. 数据加载与划分复现
# =============================================================================
df = pd.read_excel(f"{DATA}/DILI_ML_dataset_v5_final.xlsx")
log(f"[1] 数据集: {df.shape}  列示例: {list(df.columns[:4])} ... {list(df.columns[-2:])}")

X_all = df.drop(columns=["CompoundName", "SMILES", "DILI_label"])
y_all = df["DILI_label"]
names_all = df["CompoundName"]
smiles_all = df["SMILES"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEEDS["split"], stratify=y_all
)
log(f"    训练集 {X_tr.shape}  {y_tr.value_counts().to_dict()}")
log(f"    测试集 {X_te.shape}  {y_te.value_counts().to_dict()}")
log(f"    特征数 {X_all.shape[1]}")

# 与既有落盘划分核对（保证 = 之前论文所用划分）
try:
    y_tr_old = pd.read_excel(f"{DATA}/y_train.xlsx")["DILI_label"]
    same = bool((y_tr.reset_index(drop=True) == y_tr_old.reset_index(drop=True)).all())
    log(f"    与既有 y_train.xlsx 完全一致: {same}")
except Exception as e:  # pragma: no cover
    log(f"    既有划分核对失败: {e}")

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEEDS["cv"])


# =============================================================================
# 2. 三模型 5 折 CV（两套口径）
# =============================================================================
def make_lr():
    return LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=SEEDS["split"]
    )


def make_rf(**kw):
    base = dict(class_weight="balanced", random_state=SEEDS["split"], n_jobs=-1)
    base.update(kw)
    return RandomForestClassifier(**base)


def make_xgb():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEEDS["split"],
        n_jobs=-1,
    )


cv_results = {}
log("\n[2] 5 折 CV（口径 A = 训练集 444；口径 B = 全 556）")
for tag, Xc, yc in [("train444", X_tr, y_tr), ("full556", X_all, y_all)]:
    for name, estimator in [
        ("LR", make_lr()),
        ("RF_baseline", make_rf(n_estimators=500)),
        ("XGB", make_xgb()),
    ]:
        s = cross_val_score(estimator, Xc, yc, cv=cv5, scoring="roc_auc", n_jobs=-1)
        cv_results[f"{name}__{tag}"] = {
            "folds": [float(v) for v in s],
            "mean": float(s.mean()),
            "std": float(s.std()),
        }
        log(f"    {name:12s} [{tag}] mean={s.mean():.4f} ± {s.std():.4f}  folds={np.round(s,3).tolist()}")


# =============================================================================
# 3. RF 超参数搜索（原空间 + 扩展空间，均在训练集内做 CV）
# =============================================================================
log("\n[3] RF 超参数搜索（训练集内 5 折 CV，scoring=roc_auc, class_weight 固定 balanced）")
space_orig = {
    "n_estimators": [100, 300, 500, 800],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5],
}
space_ext = dict(space_orig)
space_ext["min_samples_leaf"] = [1, 2, 3, 5]

search_orig = RandomizedSearchCV(
    make_rf(), space_orig, n_iter=30, scoring="roc_auc", cv=cv5,
    random_state=SEEDS["tuning"], n_jobs=-1, verbose=0,
)
search_orig.fit(X_tr, y_tr)
log(f"    原空间:   best CV AUC={search_orig.best_score_:.4f}  params={search_orig.best_params_}")

search_ext = RandomizedSearchCV(
    make_rf(), space_ext, n_iter=60, scoring="roc_auc", cv=cv5,
    random_state=SEEDS["tuning"], n_jobs=-1, verbose=0,
)
search_ext.fit(X_tr, y_tr)
log(f"    扩展空间: best CV AUC={search_ext.best_score_:.4f}  params={search_ext.best_params_}")

# 最终模型 = 扩展空间最优（含 min_samples_leaf；class_weight 仍为 balanced）
best_params = search_ext.best_params_
final_rf = make_rf(**best_params)

# 与既有 pkl 的测试集 AUC 对照（只是记录，不作为最终值）
try:
    old_model = joblib.load(f"{DATA}/best_RF_model.pkl")
    auc_old = roc_auc_score(y_te, old_model.predict_proba(X_te)[:, 1])
    log(f"    既有 pkl 模型测试 AUC（对照）: {auc_old:.4f}")
except Exception as e:
    auc_old = None
    log(f"    既有 pkl 读取失败: {e}")


# =============================================================================
# 4. OOF 阈值选择（关键：M1 修复）
# =============================================================================
log("\n[4] 训练集 OOF 概率 → 阈值扫描（测试集不参与任何选择）")
oof_prob = cross_val_predict(
    make_rf(**best_params), X_tr, y_tr, cv=cv5, method="predict_proba", n_jobs=-1
)[:, 1]
oof_auc = roc_auc_score(y_tr, oof_prob)
log(f"    训练集 OOF AUC = {oof_auc:.4f}")

rows = []
for thr in np.round(np.arange(0.05, 0.951, 0.01), 2):
    pred = (oof_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_tr, pred).ravel()
    rows.append({
        "Threshold": float(thr),
        "Accuracy": accuracy_score(y_tr, pred),
        "Precision": precision_score(y_tr, pred, zero_division=0),
        "Recall": recall_score(y_tr, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "F1": f1_score(y_tr, pred, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    })
thr_df = pd.DataFrame(rows)
tau_star = float(thr_df.loc[thr_df["F1"].idxmax(), "Threshold"])
log(f"    OOF 最优 F1 阈值 τ* = {tau_star:.2f}  (F1={thr_df['F1'].max():.4f})")
# 参考：召回优先阈值
thr_recall = thr_df[thr_df["Recall"] >= 0.90]
tau_recall = float(thr_recall.loc[thr_recall["F1"].idxmax(), "Threshold"]) if len(thr_recall) else None
log(f"    召回≥0.90 条件下 F1 最优阈值 = {tau_recall}")


# =============================================================================
# 5. 最终模型 + 测试集一次性评估
# =============================================================================
log("\n[5] 在完整训练集上拟合最终 RF → 测试集单次评估")
final_rf.fit(X_tr, y_tr)
joblib.dump(final_rf, f"{DATA}/best_RF_model_v2.pkl")

y_prob_te = final_rf.predict_proba(X_te)[:, 1]
auc_te = roc_auc_score(y_te, y_prob_te)


def metrics_at(thr):
    pred = (y_prob_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


test_tau = metrics_at(tau_star)
test_half = metrics_at(0.5)
log(f"    测试 AUC = {auc_te:.4f}")
log(f"    @τ*={tau_star:.2f}: recall={test_tau['recall']:.3f} spec={test_tau['specificity']:.3f} "
    f"F1={test_tau['f1']:.3f} CM={test_tau['confusion_matrix']}")
log(f"    @0.50: recall={test_half['recall']:.3f} spec={test_half['specificity']:.3f} "
    f"F1={test_half['f1']:.3f} CM={test_half['confusion_matrix']}")

# bootstrap 95% CI for test AUC
rng = np.random.default_rng(SEEDS["bootstrap"])
y_te_arr, p_arr = np.asarray(y_te), y_prob_te
boot = []
for _ in range(2000):
    idx = rng.integers(0, len(y_te_arr), len(y_te_arr))
    if len(np.unique(y_te_arr[idx])) < 2:
        continue
    boot.append(roc_auc_score(y_te_arr[idx], p_arr[idx]))
auc_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
log(f"    测试 AUC 95% CI（bootstrap n=2000）= [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")

# ROC 曲线点（供 Figure1D 直接使用）
fpr, tpr, _ = roc_curve(y_te, y_prob_te)
np.save(f"{DATA}/roc_v2_fpr.npy", fpr)
np.save(f"{DATA}/roc_v2_tpr.npy", tpr)
np.save(f"{DATA}/test_prob_v2.npy", y_prob_te)
y_te.to_excel(f"{DATA}/y_test_v2.xlsx", index=False)


# =============================================================================
# 6. Scaffold split 稳健性检验（Murcko 骨架分组划分）
# =============================================================================
log("\n[6] Scaffold split（Murcko 骨架分组）稳健性检验")
scaffolds = []
for smi in smiles_all:
    m = Chem.MolFromSmiles(str(smi))
    scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None)
scaf_ser = pd.Series(scaffolds)
log(f"    唯一 Murcko 骨架数 = {scaf_ser.nunique()} / {len(scaf_ser)} 化合物")

groups = {}
for i, s in enumerate(scaffolds):
    groups.setdefault(s, []).append(i)
# 大骨架优先入训练集，直到 ~80%
order = sorted(groups.items(), key=lambda kv: -len(kv[1]))
tr_idx, te_idx = [], []
for s, idxs in order:
    if len(tr_idx) + len(idxs) <= 0.8 * len(scaffolds):
        tr_idx.extend(idxs)
    else:
        te_idx.extend(idxs)
if not te_idx:  # 退化保护
    te_idx = tr_idx[-30:]
    tr_idx = tr_idx[:-30]
log(f"    scaffold 划分: train={len(tr_idx)}  test={len(te_idx)}  "
    f"（train 覆盖 {scaf_ser.iloc[tr_idx].nunique()} 骨架）")

X_s_tr, y_s_tr = X_all.iloc[tr_idx], y_all.iloc[tr_idx]
X_s_te, y_s_te = X_all.iloc[te_idx], y_all.iloc[te_idx]
scaf_model = make_rf(**best_params).fit(X_s_tr, y_s_tr)
scaf_auc = roc_auc_score(y_s_te, scaf_model.predict_proba(X_s_te)[:, 1])
# 对照：同规模随机划分
rand_aucs = []
for rs in range(10):
    Xa, Xb, ya, yb = train_test_split(X_all, y_all, test_size=len(te_idx) / len(X_all),
                                      random_state=rs, stratify=y_all)
    m = make_rf(**best_params).fit(Xa, ya)
    rand_aucs.append(roc_auc_score(yb, m.predict_proba(Xb)[:, 1]))
log(f"    scaffold AUC = {scaf_auc:.4f}   | 随机划分 AUC(10 次) = "
    f"{np.mean(rand_aucs):.4f} ± {np.std(rand_aucs):.4f}")


# =============================================================================
# 7. 落盘
# =============================================================================
env = {
    "python": platform.python_version(),
    "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__, "xgboost": xgboost.__version__,
    "rdkit": __import__("rdkit").__version__, "joblib": joblib.__version__,
    "seeds": SEEDS,
}
json.dump(env, open(f"{DATA}/environment_v2.json", "w"), indent=2, ensure_ascii=False)

metrics = {
    "data": {
        "n_total": int(X_all.shape[0]), "n_features": int(X_all.shape[1]),
        "n_train": int(X_tr.shape[0]), "n_test": int(X_te.shape[0]),
        "train_class": {str(k): int(v) for k, v in y_tr.value_counts().items()},
        "test_class": {str(k): int(v) for k, v in y_te.value_counts().items()},
    },
    "cv": cv_results,
    "rf_search": {
        "space_orig": space_orig, "space_ext": space_ext,
        "best_cv_orig": float(search_orig.best_score_),
        "best_params_orig": search_orig.best_params_,
        "best_cv_ext": float(search_ext.best_score_),
        "best_params_ext": search_ext.best_params_,
        "auc_test_old_pkl": None if auc_old is None else float(auc_old),
    },
    "threshold_selection": {
        "oof_auc_train": float(oof_auc),
        "tau_star_f1": tau_star,
        "tau_recall_ge090": tau_recall,
    },
    "test": {
        "auc": float(auc_te), "auc_ci95_bootstrap": auc_ci,
        "at_tau_star": test_tau, "at_0.50": test_half,
    },
    "scaffold": {
        "scaffold_auc": float(scaf_auc),
        "random_split_auc_mean": float(np.mean(rand_aucs)),
        "random_split_auc_std": float(np.std(rand_aucs)),
        "n_scaffolds": int(scaf_ser.nunique()),
        "n_train": len(tr_idx), "n_test": len(te_idx),
    },
}
json.dump(metrics, open(f"{DATA}/model_metrics_v2.json", "w"), indent=2, ensure_ascii=False)
json.dump(best_params, open(f"{DATA}/RF_params_v2.json", "w"), indent=2)

thr_df.to_excel(f"{DATA}/threshold_oof_v2.xlsx", index=False)
pd.DataFrame({k: pd.Series(v["folds"]) for k, v in cv_results.items()}).to_excel(
    f"{DATA}/cv_folds_v2.xlsx", index=False)

open(f"{DATA}/20_retrain_log.txt", "w", encoding="utf-8").write("\n".join(log_lines))
log(f"\n[7] 全部产物已落盘。耗时 {time.time()-t0:.0f}s")
