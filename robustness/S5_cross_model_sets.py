# -*- coding: utf-8 -*-
"""
Step 5 — Part C: 跨模型高风险药物集合比较 (本方案核心)
产出: out/S5_scores_all_models.xlsx, out/S5_high_risk_sets.csv,
      out/S5_consensus_layers.csv, out/S5_pairwise_agreement.csv,
      figures/S5_agreement_heatmap.png, out/S5_step5.log

流程:
  1. 训练 Step 1 矩阵里的每个模型 (L1 系列为主, 全部用固定 split 的训练部分)
  2. 对同一批待评药(659 vLess+Ambiguous)打分
  3. 排序一致性(Spearman/Kendall) + Top-k% 集合 Jaccard
  4. 三分层: Consensus HIGH / Consensus LOW / Discordant  -> G4 关卡
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr, kendalltau
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

SEED = dc.SEED
log = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.append(s)

P("=" * 72)
P("Step 5 — Part C 跨模型高风险药物集合比较")
P("=" * 72)

# ======================================================================
# 1. 载入数据
# ======================================================================
ds = dc.load_dataset()                       # 556 train pool
Xtr, Xte, ytr, yte = dc.load_fixed_split()   # 固定 split 索引对齐用
raw, mat = dc.load_unknown()                 # 659
FP, DESC = dc.FP_COLS, dc.DESC_COLS

# 训练集特征 (L1)
X_train_all = ds[FP + DESC].astype(float).values
y_train_all = ds["DILI_label"].values

# 用既有固定 split 的索引复现: X_train 就是 444 子集
# 这里直接用既有 X_train/y_train 训练主模型, 保证与基线一致
Xtr_v = Xtr[FP + DESC].astype(float).values
Xte_v = Xte[FP + DESC].astype(float).values
P(f"\n[5.1] 训练 {Xtr_v.shape} / 测试 {Xte_v.shape}")

# 候选打分池 (659)
# ⚠ 坑(已修): DILI_unknown_prediction_rank.xlsx 按风险分降序,
#    DILI_unknown_ML_matrix.xlsx / DILI_unknown_compounds_raw.xlsx 按字母序,
#    三表行序完全不同 —— 必须按 CompoundName 连接, 禁止按位置 reindex。
pr = pd.read_excel(dc.DILI / "data" / "DILI_unknown_prediction_rank.xlsx")
pr["_key"] = pr["CompoundName"].astype(str).str.strip()
mat = mat.copy()
mat["_key"] = mat["CompoundName"].astype(str).str.strip()
_meta = pr[["_key", "DILI_category", "SMILES"]].drop_duplicates("_key")
cand = mat.merge(_meta, on="_key", how="left", suffixes=("", "_pr"))
assert cand["DILI_category"].notna().all(), \
    f"合并后类别缺失 {cand['DILI_category'].isna().sum()} 行 (CompoundName 未匹配)"
cand["DILI_category"] = cand["DILI_category"].astype(str).str.strip().map({
    "vLess-DILI-concern": "vLess", "Ambiguous-DILI-concern": "Ambiguous"})
assert cand["DILI_category"].notna().all(), "类别映射后仍有缺失"
X_cand = cand[FP + DESC].astype(float).values
P(f"[5.2] 候选池 {X_cand.shape}  类别 {cand['DILI_category'].value_counts().to_dict()}")

# 3D 特征(若 Step2 成功)
f3d_tr = dc.OUT / "feat3d_train.csv"
f3d_cd = dc.OUT / "feat3d_unknown.csv"
has3d = f3d_tr.exists() and f3d_cd.exists()
DESC3D = ["Asphericity","Eccentricity","InertialShapeFactor","NPR1","NPR2",
          "PMI1","PMI2","PMI3","RadiusOfGyration","SpherocityIndex","PBF"]
if has3d:
    t3d = pd.read_csv(f3d_tr, index_col=0)
    c3d = pd.read_csv(f3d_cd, index_col=0)
    P(f"[5.3] 3D 特征: train {t3d.shape} / cand {c3d.shape}  (缺失列 {t3d.isna().sum().sum()} / {c3d.isna().sum().sum()})")
else:
    P("[5.3] [WARN] 未找到 3D 特征, 跳过 F4/F5 臂")

# ======================================================================
# 2. 组装模型清单 (只取 L1 系列作跨模型比较主表)
# ======================================================================
def RF():
    return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)

models = {}
models["L1_F1_A1"] = dict(feat=FP, name="RF / Morgan only")
models["L1_F3_A1"] = dict(feat=FP + DESC, name="RF / Morgan+2D (既有基线)")
models["L1_F2_A1"] = dict(feat=DESC, name="RF / 2D desc only")

try:
    from xgboost import XGBClassifier
    def make_xgb():
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                             subsample=0.9, colsample_bytree=0.8, random_state=SEED,
                             eval_metric="logloss", n_jobs=-1)
    models["L1_F3_A2"] = dict(feat=FP + DESC, name="XGBoost / Morgan+2D", maker=make_xgb)
except Exception as e:
    P(f"[WARN] xgboost 不可用: {e}")

from sklearn.linear_model import LogisticRegression
def make_lr():
    return LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)
models["L1_F3_A3"] = dict(feat=FP + DESC, name="LogReg / Morgan+2D", maker=make_lr)

if has3d:
    models["L1_F4_A1"] = dict(feat=DESC3D, name="RF / 3D only", is3d=True)
    models["L1_F5_A1"] = dict(feat=FP + DESC + DESC3D, name="RF / 2D+3D", is3d=True)


# ======================================================================
# 3. 训练 + 打分
# ======================================================================
scores = pd.DataFrame({"LTKBID": cand.get("LTKBID", pd.Series(dtype=object)).values
                                  if "LTKBID" in cand.columns else pd.Series(dtype=object),
                       "CompoundName": cand["CompoundName"].values,
                       "DILI_category": cand["DILI_category"].values,
                       "SMILES": (cand["SMILES"].fillna(cand["SMILES_pr"])
                                  if "SMILES_pr" in cand.columns else cand["SMILES"]).values})

# 先用固定 split 训练(主模型 L1_F3_A1)
# 为保持所有模型一致, 统一用 Xtr_v / ytr 训练
meta = []
for mid, cfg in models.items():
    feat = cfg["feat"]
    if cfg.get("is3d"):
        # 3D 表索引是 ds(556) 与 cand(659) 的行序; 需按固定 split 的索引取 444 行
        # Xtr / Xte 来自既有 split, 其顺序对应 ds 的子集 —— 用 ytr 与 ds 匹配拿索引
        t3d_full = pd.concat([ds.reset_index(drop=True), t3d.reset_index(drop=True)], axis=1)
        # 找出训练行: 用既有 X_train 的 7 个描述符与 ds 全表做匹配
        key_tr = Xtr[DESC].round(6).astype(str).agg("|".join, axis=1)
        key_ds = ds[DESC].round(6).astype(str).agg("|".join, axis=1)
        pos = {k: i for i, k in enumerate(key_ds)}
        rows_tr = [pos[k] for k in key_tr if k in pos]
        if len(rows_tr) != len(Xtr):
            # 退化: 直接按 DILI_label 数量顺序对齐(不推荐, 仅兜底)
            P(f"      [WARN] 3D 行匹配 {len(rows_tr)}/{len(Xtr)}, 回退为顺序对齐")
            rows_tr = list(range(len(Xtr)))
        Xt = t3d_full.iloc[rows_tr][feat].astype(float).values
        Xc = pd.concat([cand.reset_index(drop=True), c3d.reset_index(drop=True)], axis=1)[feat].astype(float).values
    else:
        Xt = Xtr[feat].astype(float).values
        Xc = cand[feat].astype(float).values
    Xt = np.nan_to_num(Xt); Xc = np.nan_to_num(Xc)

    m = cfg.get("maker", RF)()
    m.fit(Xt, ytr)
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(Xc)[:, 1]
    else:
        p = m.decision_function(Xc)
    scores[mid] = p
    meta.append(dict(model_id=mid, name=cfg["name"], n_feat=len(feat),
                     score_min=float(p.min()), score_max=float(p.max())))
    P(f"[5.4] {mid:12s} {cfg['name']:30s} n_feat={len(feat):5d}  "
      f"score [{p.min():.4f}, {p.max():.4f}]")

meta_df = pd.DataFrame(meta)
score_cols = list(models.keys())
scores.to_excel(dc.OUT / "S5_scores_all_models.xlsx", index=False)

# ======================================================================
# 4. 排序一致性
# ======================================================================
P("\n[5.5] 模型两两排序一致性")
sp = pd.DataFrame(index=score_cols, columns=score_cols, dtype=float)
kd = pd.DataFrame(index=score_cols, columns=score_cols, dtype=float)
for i in score_cols:
    for j in score_cols:
        sp.loc[i, j] = spearmanr(scores[i], scores[j]).statistic
        kd.loc[i, j] = kendalltau(scores[i], scores[j]).statistic
sp.to_csv(dc.OUT / "S5_pairwise_spearman.csv", encoding="utf-8-sig")
P("      Spearman ρ:")
P(sp.round(4).to_string())

# ======================================================================
# 5. Top-k% 集合 + Jaccard
# ======================================================================
def high_set(p, k_pct):
    thr = np.quantile(p, 1 - k_pct / 100)
    return set(np.where(p >= thr)[0])

for k in (5, 10):
    P(f"\n[5.6] Top-{k}% 集合 Jaccard")
    Hs = {mid: high_set(scores[mid].values, k) for mid in score_cols}
    jac = pd.DataFrame(index=score_cols, columns=score_cols, dtype=float)
    for i in score_cols:
        for j in score_cols:
            u = len(Hs[i] | Hs[j]); jac.loc[i, j] = len(Hs[i] & Hs[j]) / u if u else np.nan
    P(jac.round(3).to_string())
    jac.to_csv(dc.OUT / f"S5_jaccard_top{k}.csv", encoding="utf-8-sig")

    # 三分层: 严格定义 (全票 HIGH 在本数据里为 0 -> 用"强共识"阈值 n-1)
    cnt = np.zeros(len(scores))
    for mid in score_cols:
        cnt += np.isin(np.arange(len(scores)), list(Hs[mid]))
    n_models = len(score_cols)
    n_full = int((cnt == n_models).sum())
    hi_thr = n_models - 1 if n_full == 0 else n_models   # 无全票时用 n-1 作"强共识"
    layers = np.where(cnt >= hi_thr, "Consensus_HIGH",
             np.where(cnt == 0, "Consensus_LOW", "Discordant"))
    scores[f"n_high_top{k}"] = cnt
    scores[f"layer_top{k}"] = layers
    P(f"      分层(阈值: HIGH≥{hi_thr}票, 全票数={n_full}): {pd.Series(layers).value_counts().to_dict()}")

# 以 Top-10% 作为主分层
k_main = 10
scores["n_high"] = scores[f"n_high_top{k_main}"]
scores["layer"] = scores[f"layer_top{k_main}"]

layer_summary = scores.groupby("layer").agg(
    n=("CompoundName", "size"),
    n_vLess=("DILI_category", lambda s: (s == "vLess").sum()),
    n_amb=("DILI_category", lambda s: (s == "Ambiguous").sum()),
    mean_n_high=("n_high", "mean"),
).reset_index()
layer_summary.to_csv(dc.OUT / "S5_consensus_layers.csv", index=False, encoding="utf-8-sig")
P("\n[5.7] 共识分层汇总:")
P(layer_summary.to_string(index=False))

# G4
n_disc = int((scores["layer"] == "Discordant").sum())
P("\n" + "=" * 72)
P(f"[G4] Discordant 集合 n = {n_disc}   (要求 ≥15)")
P(f"[G4] 判定: {'通过' if n_disc >= 15 else '未通过 - 改为模型高度一致的正向结论'}")

# 保存关键列
scores[["CompoundName", "DILI_category"] + score_cols + ["n_high", "layer"]].to_csv(
    dc.OUT / "S5_high_risk_sets.csv", index=False, encoding="utf-8-sig")

# ======================================================================
# 6. 分歧药排序(按跨模型分数标准差)
# ======================================================================
sd_col = scores[score_cols].std(axis=1)
scores["score_sd"] = sd_col
disc = scores[scores["layer"] == "Discordant"].sort_values("score_sd", ascending=False)
disc.to_csv(dc.OUT / "S5_discordant_ranked.csv", index=False, encoding="utf-8-sig")
P(f"\n[5.8] Discordant 按 score_sd 排序 Top-15:")
P(disc[["CompoundName", "DILI_category", "score_sd", "n_high"]].head(15).to_string(index=False))

(dc.OUT / "S5_step5.log").write_text("\n".join(log), encoding="utf-8")
P("\nStep 5 完成。")
