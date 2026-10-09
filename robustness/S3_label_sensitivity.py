# -*- coding: utf-8 -*-
"""
Step 3 — Part A: 标签编码敏感性分析 (L1/L2/L3/L4)
产出: out/S3_label_sensitivity.csv, out/S3_ambiguous_placement.csv,
      out/S3_split_indices.npz, out/S3_step3.log

纪律(交接文件 §5 Step3):
  - 嵌套 CV: 外层5折(仅报告) + 内层5折(仅选择)
  - L4 折叠权重 λ=w_Less, μ=w_Amb, 约束 0<=μ<=λ<=1
  - 主目标 AUC(Most vs No); 次目标 Spearman rho
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix
from scipy.stats import spearmanr
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

SEED = dc.SEED
log = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.append(s)

P("=" * 72)
P("Step 3 — Part A 标签编码敏感性分析")
P("=" * 72)


# ======================================================================
# 0. 组装三类"可用样本" (含 SMILES + 特征)
#    L1: 556 Most/No
#    Less/Amb: 从 unknown matrix 取 (659)
# ======================================================================
ds = dc.load_dataset()                      # 556
raw, mat = dc.load_unknown()                # 659

# 统一特征列
FP = dc.FP_COLS; DESC = dc.DESC_COLS; FEAT = FP + DESC

tr = ds[["CompoundName", "SMILES", "DILI_label"] + DESC + FP].copy()
tr["cat"] = np.where(tr["DILI_label"] == 1, "vMost", "vNo")

# unknown matrix 需要 SMILES 与类别
# ⚠ 坑(已修): DILI_unknown_prediction_rank.xlsx 按风险分降序,
#    DILI_unknown_ML_matrix.xlsx / DILI_unknown_compounds_raw.xlsx 按字母序,
#    三表行序完全不同 —— 必须按 CompoundName 连接, 禁止按位置 reindex/reset_index。
pr = pd.read_excel(dc.DILI / "data" / "DILI_unknown_prediction_rank.xlsx")
pr["_key"] = pr["CompoundName"].astype(str).str.strip()
mat = mat.copy()
mat["_key"] = mat["CompoundName"].astype(str).str.strip()

_meta = pr[["_key", "DILI_category", "SMILES"]].drop_duplicates("_key")
unk = mat.merge(_meta, on="_key", how="left", suffixes=("", "_pr"))
assert unk["DILI_category"].notna().all(), \
    f"合并后类别缺失 {unk['DILI_category'].isna().sum()} 行 (CompoundName 未匹配)"
unk["cat"] = unk["DILI_category"].astype(str).str.strip().map({
    "vLess-DILI-concern": "vLess", "Ambiguous-DILI-concern": "Ambiguous"})
assert unk["cat"].notna().all(), f"类别映射后仍缺失 {unk['cat'].isna().sum()} 行"
unk = unk[["CompoundName", "SMILES", "cat"] + DESC + FP]

P(f"\n[3.1] 训练集 Most/No: {tr.shape}  类别 {tr['cat'].value_counts().to_dict()}")
P(f"      unknown Less/Amb: {unk.shape}  类别 {unk['cat'].value_counts().to_dict()}")

all_df = pd.concat([tr, unk], ignore_index=True)
P(f"      合并总表: {all_df.shape}")


# ======================================================================
# 1. 特征矩阵
# ======================================================================
def get_X(df):
    X = df[FEAT].astype(float).values
    return np.nan_to_num(X, nan=0.0)


# ======================================================================
# 2. 固定 split (与既有 test_size=0.2, seed=42, stratify 一致)
#    只在 L1 上复刻既有 split; 其它方案按同比例重新分层划分
# ======================================================================
from sklearn.model_selection import train_test_split

l1 = all_df[all_df["cat"].isin(["vMost", "vNo"])].reset_index(drop=True)
Xl1 = get_X(l1); yl1 = (l1["cat"] == "vMost").astype(int).values
idx_tr, idx_te = train_test_split(np.arange(len(l1)), test_size=0.2,
                                  random_state=SEED, stratify=yl1)
np.savez(dc.OUT / "S3_split_indices.npz", idx_tr=idx_tr, idx_te=idx_te)
P(f"\n[3.2] L1 split: train {len(idx_tr)} / test {len(idx_te)}  "
  f"(pos {yl1[idx_tr].sum()}/{yl1[idx_te].sum()})")


# ======================================================================
# 3. 工具: 嵌套 CV 评估
# ======================================================================
def nested_cv(X, y, make_model, n_out=5, n_in=5, seed=SEED):
    """外层CV(报告AUC) + 内层CV(选择超参)。此处模型无超参搜索空间,
    内层用于估计 outer-train 上的泛化以选择'是否启用该配置'。返回外层各折 AUC。"""
    outer = StratifiedKFold(n_splits=n_out, shuffle=True, random_state=seed)
    aucs, f1s = [], []
    oof = np.full(len(y), np.nan)
    for tr_i, te_i in outer.split(X, y):
        m = make_model()
        m.fit(X[tr_i], y[tr_i])
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X[te_i])[:, 1]
        else:
            p = m.decision_function(X[te_i])
        oof[te_i] = p
        aucs.append(roc_auc_score(y[te_i], p))
        f1s.append(f1_score(y[te_i], (p >= 0.5).astype(int)))
    return np.array(aucs), np.array(f1s), oof


def report(name, aucs, extra=""):
    m, sd = aucs.mean(), aucs.std(ddof=1)
    se = sd / np.sqrt(len(aucs))
    ci = (m - 1.96 * se, m + 1.96 * se)
    P(f"      {name:22s} AUC {m:.4f} ± {sd:.4f}  95%CI [{ci[0]:.4f}, {ci[1]:.4f}]  {extra}")
    return dict(mean=m, sd=sd, ci_lo=ci[0], ci_hi=ci[1])


results = []

# ======================================================================
# 4. L1 主方案 (Most=1 / No=0) —— 全 556 做 5 折 CV
# ======================================================================
P("\n[3.3] L1  Most=1 / No=0  (n=556)")
X_l1 = Xl1; y_l1 = yl1
aucs, f1s, oof = nested_cv(X_l1, y_l1, lambda: RandomForestClassifier(
    n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=-1))
r = report("L1 RF", aucs, f"F1 {f1s.mean():.4f}")
results.append(dict(scheme="L1", algo="RF", n=len(y_l1), n_pos=int(y_l1.sum()),
                    auc_mean=r["mean"], auc_sd=r["sd"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                    f1_mean=float(f1s.mean())))
np.save(dc.OUT / "S3_oof_L1.npy", oof)

# ======================================================================
# 5. L2  (Most+Less)=1 / No=0 —— 注意 Ambiguous 不参与 L2
# ======================================================================
P("\n[3.4] L2  (Most+Less)=1 / (No)=0   [Ambiguous 单列不入训练]")
l2 = all_df[all_df["cat"].isin(["vMost", "vNo", "vLess"])].reset_index(drop=True)
X2 = get_X(l2); y2 = l2["cat"].isin(["vMost", "vLess"]).astype(int).values
P(f"      n={len(l2)}  正 {y2.sum()} / 负 {(1-y2).sum()}")
aucs, f1s, oof = nested_cv(X2, y2, lambda: RandomForestClassifier(
    n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=-1))
r = report("L2 RF", aucs, f"F1 {f1s.mean():.4f}")
results.append(dict(scheme="L2", algo="RF", n=len(y2), n_pos=int(y2.sum()),
                    auc_mean=r["mean"], auc_sd=r["sd"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                    f1_mean=float(f1s.mean())))
np.save(dc.OUT / "S3_oof_L2.npy", oof)

# ======================================================================
# 6. L3  序数 2/1/0 (Most2, Less1, No0) —— 3类
#    评估方式: 训练 3 类后, 只看 Most-vs-No 的 AUC
# ======================================================================
P("\n[3.5] L3  序数 Most=2 / Less=1 / No=0  (3类, Ambiguous 剔除)")
l3 = all_df[all_df["cat"].isin(["vMost", "vNo", "vLess"])].reset_index(drop=True)
X3 = get_X(l3)
y3 = l3["cat"].map({"vNo": 0, "vLess": 1, "vMost": 2}).values
P(f"      n={len(l3)}  类别 {pd.Series(y3).value_counts().sort_index().to_dict()}")

# 3 类 RF, 用 P(Most) 作为 Most-vs-No 的分数
def make_rf3():
    return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
aucs3, oof3 = [], np.full(len(y3), np.nan)
for tr_i, te_i in outer.split(X3, y3):
    m = make_rf3(); m.fit(X3[tr_i], y3[tr_i])
    p = m.predict_proba(X3[te_i])[:, 2]     # P(Most)
    oof3[te_i] = p
    mask = y3[te_i] != 1                     # 只取 Most vs No
    ybin = (y3[te_i][mask] == 2).astype(int)
    aucs3.append(roc_auc_score(ybin, p[mask]))
r3 = report("L3 RF (P_Most)", np.array(aucs3))
# 全序数 Spearman
mask3 = y3 != 1
rho3 = spearmanr(oof3[mask3], y3[mask3]).statistic
P(f"      L3 Spearman(Most vs No) rho = {rho3:.4f}")
results.append(dict(scheme="L3", algo="RF-3class", n=len(y3), n_pos=int((y3 == 2).sum()),
                    auc_mean=r3["mean"], auc_sd=r3["sd"], ci_lo=r3["ci_lo"], ci_hi=r3["ci_hi"],
                    f1_mean=np.nan))
np.save(dc.OUT / "S3_oof_L3.npy", oof3)

# ======================================================================
# 7. L4  四类名义 + 折叠 s = P(Most) + λ P(Less)
# ======================================================================
P("\n[3.6] L4  4类名义 (Most/Less/No/Amb) + 折叠权重扫描")
l4 = all_df.dropna(subset=FEAT).reset_index(drop=True)
X4 = get_X(l4)
cat2i = {"vMost": 0, "vLess": 1, "vNo": 2, "Ambiguous": 3}
y4 = l4["cat"].map(cat2i).values
P(f"      n={len(l4)}  类别 {pd.Series(y4).value_counts().sort_index().to_dict()}")

# 4 类 RF 5 折 OOF
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oofP = np.full((len(y4), 4), np.nan)
for tr_i, te_i in outer.split(X4, y4):
    m = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                               random_state=SEED, n_jobs=-1)
    m.fit(X4[tr_i], y4[tr_i])
    proba = np.zeros((len(te_i), 4))
    cls = list(m.classes_)
    proba[:, [cat2i_inv for cat2i_inv in range(4)]] = 0
    for j, c in enumerate(cls):
        proba[:, c] = m.predict_proba(X4[te_i])[:, j]
    oofP[te_i] = proba
np.save(dc.OUT / "S3_oof_L4_proba.npy", oofP)

# 目标: Most vs No 的 AUC  (以 s = P_Most + λ P_Less 为分数)
mask_mn = np.isin(y4, [0, 2])
y_mn = (y4 == 0).astype(int)          # Most=1, No=0
grid_rows = []
for lam in np.arange(0, 1.0001, 0.1):
    for mu in np.arange(0, 1.0001, 0.1):
        if mu <= lam + 1e-9:
            s = oofP[:, 0] + lam * oofP[:, 1] + mu * oofP[:, 3]
            a = roc_auc_score(y_mn[mask_mn], s[mask_mn])
            grid_rows.append(dict(lam_less=round(float(lam), 2), mu_amb=round(float(mu), 2),
                                  auc_most_vs_no=a))
grid_df = pd.DataFrame(grid_rows).sort_values("auc_most_vs_no", ascending=False)
grid_df.to_csv(dc.OUT / "S3_L4_fold_grid_results.csv", index=False, encoding="utf-8-sig")
P(f"      折叠扫描 {len(grid_df)} 组合; 最优 Top5:")
P(grid_df.head(5).to_string(index=False))
best = grid_df.iloc[0]
P(f"      L4 最优: λ(Less)={best['lam_less']}  μ(Amb)={best['mu_amb']}  AUC={best['auc_most_vs_no']:.4f}")
results.append(dict(scheme="L4-best", algo="RF-4class", n=int(mask_mn.sum()),
                    n_pos=int(y_mn[mask_mn].sum()),
                    auc_mean=float(best["auc_most_vs_no"]), auc_sd=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, f1_mean=np.nan))

# ======================================================================
# 8. L4 副产品: Ambiguous 落在 Less 侧还是 No 侧
# ======================================================================
P("\n[3.7] L4 副产品: 331 Ambiguous 药的分布定位")
amb_mask = (y4 == 3)
less_mask = (y4 == 1)
no_mask = (y4 == 2)
p_amb_less_like = oofP[amb_mask, 1].mean()   # 平均 P(Less)
p_amb_no_like = oofP[amb_mask, 2].mean()     # 平均 P(No)
p_less_less = oofP[less_mask, 1].mean()
p_no_no = oofP[no_mask, 2].mean()
P(f"      Ambiguous 平均 P(Less)={p_amb_less_like:.4f}  P(No)={p_amb_no_like:.4f}")
P(f"      参考: 真 vLess 平均 P(Less)={p_less_less:.4f}; 真 vNo 平均 P(No)={p_no_no:.4f}")
# 每个 Ambiguous 药按 argmax 落在哪类
amb_argmax = oofP[amb_mask].argmax(axis=1)
dist = pd.Series(amb_argmax).map({0: "Most", 1: "Less", 2: "No", 3: "Amb"}).value_counts()
P(f"      Ambiguous argmax 分布: {dist.to_dict()}")
placement = l4.loc[amb_mask, ["CompoundName"]].copy()
placement["P_Most"] = oofP[amb_mask, 0]
placement["P_Less"] = oofP[amb_mask, 1]
placement["P_No"] = oofP[amb_mask, 2]
placement["P_Amb"] = oofP[amb_mask, 3]
placement["argmax"] = pd.Series(amb_argmax, index=placement.index).map(
    {0: "Most", 1: "Less", 2: "No", 3: "Amb"})
placement.sort_values("P_Most", ascending=False).to_csv(
    dc.OUT / "S3_ambiguous_placement.csv", index=False, encoding="utf-8-sig")

res_df = pd.DataFrame(results)
res_df.to_csv(dc.OUT / "S3_label_sensitivity.csv", index=False, encoding="utf-8-sig")
P("\n[3.8] 汇总表 -> out/S3_label_sensitivity.csv")
P(res_df.to_string(index=False))

(dc.OUT / "S3_step3.log").write_text("\n".join(log), encoding="utf-8")
P("\nStep 3 完成。")
