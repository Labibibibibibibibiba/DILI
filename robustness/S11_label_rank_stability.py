# -*- coding: utf-8 -*-
"""
S11 — V2.2 核心 8 药在「全标签方案 × 全算法 × 全特征空间」下的排名稳健性检验

回答的问题（论文稳健性附录用）:
  V2.2 方案选定的候选药(CPZ/SMX/CAB/MBZ)与阴性对照(OXY/TOL/MET),
  其排名是否对「标签编码方案」稳健?

覆盖:
  L1  Most=1/No=0            (二分类, Ambiguous 剔除)
  L2  (Most+Less)=1/No=0     (二分类, Ambiguous 剔除)
  L3  Most2/Less1/No0        (3类, 用 P(Most) 作 Most-vs-No 分数)
  L4  4类名义 + 折叠 s = P_Most + λ·P_Less + μ·P_Amb  (λ=0.6, μ=0.5)

  算法: RF(A1) / XGB(A2) / LR(A3)
  特征: F1(Morgan) / F2(7 desc) / F3(F1+F2)

统一口径: 训练集 = 全部可用样本(与 S3 的 CV 口径一致, 非固定 split),
          对 659 unknown 打分, 计算排名(1=最危险)。

只读 D:\DILI, 只写 D:\DILI_1。
产出: out/S11_label_rank_stability.csv  (长表: 药 × 方案 × 算法 × 特征)
      out/S11_rank_summary.csv         (宽表: 汇总)
      out/S11_step11.log
"""
import sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

SEED = dc.SEED
log = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.append(s)

P("=" * 78)
P("S11 — V2.2 核心 8 药 标签方案稳健性排名")
P("=" * 78)

# ======================================================================
# 0. 组装数据 (与 S3 完全一致的口径)
# ======================================================================
ds = dc.load_dataset()          # 556 Most/No, 含 SMILES + 特征
raw, mat = dc.load_unknown()    # 659 unknown
FP, DESC = dc.FP_COLS, dc.DESC_COLS
FEAT = FP + DESC

tr = ds[["CompoundName", "SMILES"] + DESC + FP].copy()
tr["cat"] = np.where(ds["DILI_label"].values == 1, "vMost", "vNo")

unk = mat.copy()
# ⚠ 坑: DILI_unknown_prediction_rank / ML_matrix / compounds_raw 三表行序完全不同
#    (prediction_rank 按风险分降序; ML_matrix 与 compounds_raw 按字母序)
#    必须按 CompoundName 连接, 禁止按位置 reindex —— 否则类别会错位 1~N 行。
pr = pd.read_excel(dc.DILI / "data" / "DILI_unknown_prediction_rank.xlsx")
pr["_key"] = pr["CompoundName"].astype(str).str.strip()
unk["_key"] = unk["CompoundName"].astype(str).str.strip()
key_map = dict(zip(pr["_key"], pr["DILI_category"]))
unk["cat_raw"] = unk["_key"].map(key_map)
# 保底: 若 SMILES 缺失, 从 pr 按名取
if "SMILES" not in unk.columns or unk["SMILES"].isna().any():
    smi_map = dict(zip(pr["_key"], pr["SMILES"]))
    unk["SMILES"] = unk["_key"].map(smi_map)
unk["cat"] = unk["cat_raw"].astype(str).str.strip().map(
    {"vLess-DILI-concern": "vLess", "Ambiguous-DILI-concern": "Ambiguous"})
assert unk["cat"].notna().all(), f"类别映射后仍有缺失: {unk['cat'].isna().sum()}"
unk = unk[["CompoundName", "SMILES", "cat"] + DESC + FP]

all_df = pd.concat([tr, unk], ignore_index=True)
P(f"[11.1] 训练池 556 ({tr['cat'].value_counts().to_dict()})")
P(f"       候选池 659 ({unk['cat'].value_counts().to_dict()})")
P(f"       合并 {all_df.shape}")


def get_X(df, cols):
    X = df[cols].astype(float).values
    return np.nan_to_num(X, nan=0.0)


def make_rf(): return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                             random_state=SEED, n_jobs=-1)
def make_lr(): return LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)
def make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                         subsample=0.9, colsample_bytree=0.8, random_state=SEED,
                         eval_metric="logloss", n_jobs=-1)

ALGOS = {"RF": make_rf, "XGB": make_xgb, "LR": make_lr}
FEATS = {"F1": FP, "F2": DESC, "F3": FP + DESC}

# ======================================================================
# 1. 逐 (标签方案 × 算法 × 特征) 训练 → 对 659 打分
# ======================================================================
records = []   # 长表
cand_names = unk["CompoundName"].values

def score_and_rank(scheme, algo, fs, y, train_df, score_fn):
    """train_df 为用于训练的完整表(含 cat); score_fn(proba)->1d 分数; 返回 659 排名。"""
    cols = FEATS[fs]
    Xtr = get_X(train_df, cols)
    Xcand = get_X(unk, cols)
    m = ALGOS[algo]()
    m.fit(Xtr, y)
    proba = m.predict_proba(Xcand)
    s = score_fn(m, proba)
    rank = pd.Series(s).rank(ascending=False, method="min").astype(int).values
    for nm, sc, rk in zip(cand_names, s, rank):
        records.append(dict(scheme=scheme, algo=algo, feat=fs,
                            CompoundName=nm, score=float(sc), rank=int(rk)))
    return rank


def one_prob(m, proba):
    """二分类: 取 P(正类=1), 兼容 classes_ 顺序。"""
    cls = list(m.classes_)
    j = cls.index(1) if 1 in cls else -1
    return proba[:, j]

# 多分类统一用整数编码, 0=vMost(正类) / 1=vLess / 2=vNo / 3=Ambiguous
CAT2I = {"vMost": 0, "vLess": 1, "vNo": 2, "Ambiguous": 3}

def pmax_prob(m, proba):
    """多分类: 取 P(Most)。联合 CAT2I 使用, Most 恒为 0 类。"""
    cls = list(m.classes_)
    j = cls.index(0) if 0 in cls else 0
    return proba[:, j]


P("\n[11.2] 开始 4 方案 × 3 算法 × 3 特征 = 36 组合")

# ---------- L1: Most=1 / No=0 ----------
d1 = all_df[all_df["cat"].isin(["vMost", "vNo"])].reset_index(drop=True)
y1 = (d1["cat"] == "vMost").astype(int).values
for algo, fs in itertools.product(ALGOS, FEATS):
    score_and_rank("L1", algo, fs, y1, d1, one_prob)
P(f"       L1 完成 (n={len(y1)}, pos={y1.sum()})")

# ---------- L2: (Most+Less)=1 / No=0 ----------
d2 = all_df[all_df["cat"].isin(["vMost", "vNo", "vLess"])].reset_index(drop=True)
y2 = d2["cat"].isin(["vMost", "vLess"]).astype(int).values
for algo, fs in itertools.product(ALGOS, FEATS):
    score_and_rank("L2", algo, fs, y2, d2, one_prob)
P(f"       L2 完成 (n={len(y2)}, pos={y2.sum()})")

# ---------- L3: 3 类 Most/Less/No, 分数 = P(Most) ----------
d3 = all_df[all_df["cat"].isin(["vMost", "vNo", "vLess"])].reset_index(drop=True)
y3 = d3["cat"].map({"vMost": 0, "vLess": 1, "vNo": 2}).values  # int 编码
for algo, fs in itertools.product(ALGOS, FEATS):
    score_and_rank("L3", algo, fs, y3, d3, pmax_prob)
P(f"          L3 完成 (n={len(y3)}, {pd.Series(y3).value_counts().sort_index().to_dict()})")

# ---------- L4: 4 类 + 折叠 s = P_Most + 0.6*P_Less + 0.5*P_Amb ----------
d4 = all_df.dropna(subset=FEAT).reset_index(drop=True)
y4 = d4["cat"].map(CAT2I).values
LAM, MU = 0.6, 0.5
def l4_score(m, proba):
    cls = list(m.classes_)
    def gp(i):
        return proba[:, cls.index(i)] if i in cls else 0.0
    return gp(0) + LAM * gp(1) + MU * gp(3)   # P_Most + λP_Less + μP_Amb
for algo, fs in itertools.product(ALGOS, FEATS):
    score_and_rank("L4", algo, fs, y4, d4, l4_score)
P(f"       L4 完成 (n={len(y4)}, {pd.Series(y4).value_counts().to_dict()})")

long = pd.DataFrame(records)
long.to_csv(dc.OUT / "S11_label_rank_stability.csv", index=False, encoding="utf-8-sig")
P(f"\n[11.3] 长表 -> out/S11_label_rank_stability.csv  ({len(long)} 行)")

# ======================================================================
# 2. 汇总: 目标 8 药 × 4 方案
# ======================================================================
TARGETS = {
    "Chlorpromazine hydrochloride": ("CPZ", "阳性·池内桥接"),
    "Diclofenac":                   ("DCF", "阳性·池外基准"),
    "Sulfamethizole":               ("SMX", "高风险候选"),
    "Cabotegravir sodium":          ("CAB", "高风险候选"),
    "Mebendazole":                  ("MBZ", "高风险候选"),
    "Oxybutynin chloride":          ("OXY", "阴性 TierA"),
    "Tolterodine tartrate":         ("TOL", "阴性 TierA"),
    "Metformin hydrochloride":      ("MET", "阴性 TierB"),
    "Pioglitazone hydrochloride":   ("PIO", "不纳入(审计)"),
    "Entacapone":                   ("ENT", "不纳入(审计)"),
}
N = 659
rows = []
for full, (abbr, role) in TARGETS.items():
    sub = long[long["CompoundName"] == full]
    if len(sub) == 0:
        rows.append(dict(drug=abbr, role=role, present=False))
        continue
    n_models = len(sub)
    rec = dict(drug=abbr, role=role, present=True, n_models=n_models,
               rank_mean=sub["rank"].mean(), rank_min=sub["rank"].min(),
               rank_max=sub["rank"].max(),
               pct_top10=100.0 * (sub["rank"] <= int(N * 0.10)).mean(),
               pct_top20=100.0 * (sub["rank"] <= int(N * 0.20)).mean())
    for sch in ["L1", "L2", "L3", "L4"]:
        s = sub[sub["scheme"] == sch]
        rec[f"{sch}_rank_mean"] = s["rank"].mean() if len(s) else np.nan
        rec[f"{sch}_rank_min"] = s["rank"].min() if len(s) else np.nan
        rec[f"{sch}_rank_max"] = s["rank"].max() if len(s) else np.nan
    rows.append(rec)

summ = pd.DataFrame(rows)
summ.to_csv(dc.OUT / "S11_rank_summary.csv", index=False, encoding="utf-8-sig")

P("\n[11.4] 汇总表 -> out/S11_rank_summary.csv")
P("\n=== 各药跨 36 模型(4方案×3算法×3特征)的排名 ===")
P(f"{'药':6s}{'角色':16s}{'均':>7s}{'最小':>6s}{'最大':>6s}{'Top10%':>8s}{'L1均':>7s}{'L2均':>7s}{'L3均':>7s}{'L4均':>7s}")
for r in rows:
    if not r.get("present"):
        P(f"{r['drug']:6s}{r['role']:16s}   未在 659 库中")
        continue
    P(f"{r['drug']:6s}{r['role']:16s}{r['rank_mean']:>7.1f}{r['rank_min']:>6d}{r['rank_max']:>6d}"
      f"{r['pct_top10']:>7.0f}%{r['L1_rank_mean']:>7.1f}{r['L2_rank_mean']:>7.1f}"
      f"{r['L3_rank_mean']:>7.1f}{r['L4_rank_mean']:>7.1f}")

P("\n[11.5] 每个标签方案内, 4 核心药 vs 3 阴性药的分离度")
for sch in ["L1", "L2", "L3", "L4"]:
    core = summ[summ["drug"].isin(["CPZ", "SMX", "CAB", "MBZ"])][f"{sch}_rank_max"].max()
    neg = summ[summ["drug"].isin(["OXY", "TOL", "MET"])][f"{sch}_rank_min"].min()
    P(f"    {sch}: 核心药最差排名 {core:.0f}  vs  阴性药最好排名 {neg:.0f}  →  间隙 {neg-core:.0f}")

(dc.OUT / "S11_step11.log").write_text("\n".join(log), encoding="utf-8")
P("\nS11 完成。")
