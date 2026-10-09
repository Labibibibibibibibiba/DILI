# -*- coding: utf-8 -*-
"""
Step 1 — 定义模型矩阵并落表 (只读 D:\DILI)
产出: out/S1_model_matrix.csv
       out/S1_dataset_loaders.py  (供后续步骤 import 的共享数据加载器)

设计原则:
  - 标签方案 L1..L4 / 描述符空间 F1..F6 / 算法 A1..A3
  - 只做"有科学问题的组合", 不做全交叉
  - 所有类别筛选用规范化大小写(见交接文件 §2.2)
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

DILI = Path(r"D:\DILI")
OUT = Path(r"D:\DILI_1\out")
OUT.mkdir(parents=True, exist_ok=True)

FDA = DILI / "data" / "Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx"
DATASET = DILI / "data" / "DILI_ML_dataset_v5_final.xlsx"
X_TRAIN = DILI / "data" / "X_train.xlsx"
X_TEST = DILI / "data" / "X_test.xlsx"
Y_TRAIN = DILI / "data" / "y_train.xlsx"
Y_TEST = DILI / "data" / "y_test.xlsx"
UNKNOWN_RAW = DILI / "data" / "DILI_unknown_compounds_raw.xlsx"
UNKNOWN_MATRIX = DILI / "data" / "DILI_unknown_ML_matrix.xlsx"

log = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.append(s)


# ======================================================================
# 0. 读 FDA 原始表, 规范化类别
# ======================================================================
def load_fda_categories():
    fda = pd.read_excel(FDA, header=1)
    c = fda["vDILI-Concern"].astype(str).str.strip()
    # 统一大小写变体: vMOST -> vMost, vNo-...-Concern -> vNo-...-concern
    c = c.str.replace("vMOST", "vMost", regex=False)
    c = c.str.replace("Concern", "concern", regex=False)
    fda = fda.assign(_cat=c)
    mapping = {
        "vMost-DILI-concern": "vMost",
        "vLess-DILI-concern": "vLess",
        "vNo-DILI-concern": "vNo",
        "Ambiguous-DILI-concern": "Ambiguous",
    }
    fda["DILI_category"] = fda["_cat"].map(mapping)
    unmapped = fda.loc[fda["DILI_category"].isna(), "_cat"].value_counts()
    if len(unmapped):
        P("[WARN] 未映射的类别值:", dict(unmapped))
    return fda


P("=" * 72)
P("Step 1 — 模型矩阵落表")
P("=" * 72)

fda = load_fda_categories()
P("\n[1.1] FDA 规范化后类别分布:")
for k, v in fda["DILI_category"].value_counts().items():
    P(f"      {k:12s} {v}")


# ======================================================================
# 1. 载入训练集 (用于统计各标签方案的 n)
# ======================================================================
ds = pd.read_excel(DATASET)
P(f"\n[1.2] DILI_ML_dataset_v5_final shape = {ds.shape}")
P(f"      columns 前 12 个: {list(ds.columns[:12])}")

# 训练集 556: Most/No + SMILES 可得
n_l1_pos = int((ds["DILI_label"] == 1).sum())
n_l1_neg = int((ds["DILI_label"] == 0).sum())
P(f"      L1: Most {n_l1_pos} / No {n_l1_neg} / 合计 {n_l1_pos+n_l1_neg}")

# 未知队列
unk = pd.read_excel(UNKNOWN_RAW)
n_less = int((unk["DILI_category"].astype(str).str.strip() == "vLess-DILI-concern").sum())
n_amb = int((unk["DILI_category"].astype(str).str.strip() == "Ambiguous-DILI-concern").sum())
P(f"      unknown raw: vLess {n_less} / Ambiguous {n_amb} / 合计 {len(unk)}")

unk_mat = pd.read_excel(UNKNOWN_MATRIX)
P(f"      unknown matrix (SMILES 可得后): {unk_mat.shape}")


# ======================================================================
# 2. 各类别在"有 SMILES 且有特征"意义下的可用样本数
#     训练集 556 已满足; Less/Amb 需要从 unknown matrix 拿
# ======================================================================
n_l2_pos = n_l1_pos + len(unk_mat)   # 粗略: Most + (Less+Amb 里的全部)
n_l2_neg = n_l1_neg
P(f"\n[1.3] 标签方案可用样本(上界估计, 精确值由 Step 3 落地):")
P(f"      L1  n = {n_l1_pos + n_l1_neg}  ({n_l1_pos}+ / {n_l1_neg}-)")
P(f"      L2  n = {n_l1_pos + len(unk_mat) + n_l1_neg}  ({n_l1_pos+len(unk_mat)}+ / {n_l1_neg}-)")
P(f"      L3  3 类: Most {n_l1_pos} / Less {n_less}(上界) / No {n_l1_neg}")
P(f"      L4  4 类: Most {n_l1_pos} / Less {n_less} / No {n_l1_neg} / Amb {n_amb}")


# ======================================================================
# 3. 定义矩阵: 只做有科学问题的组合
# ======================================================================
rows = []

def add(mid, label, feat, algo, npos, nneg, notes):
    rows.append(dict(model_id=mid, label_scheme=label, feature_set=feat,
                     algorithm=algo, n_train_pos=npos, n_train_neg=nneg, notes=notes))

# ---- 主线: L1 (主标签) × F1/F2/F3 × A1 ----
add("L1_F1_A1", "L1", "F1", "A1", n_l1_pos, n_l1_neg, "基线复刻(RF, 仅Morgan)")
add("L1_F3_A1", "L1", "F3", "A1", n_l1_pos, n_l1_neg, "★既有模型实际配置(RF, Morgan+7desc)")
add("L1_F2_A1", "L1", "F2", "A1", n_l1_pos, n_l1_neg, "仅7个2D描述符")
add("L1_F4_A1", "L1", "F4", "A1", n_l1_pos, n_l1_neg, "★仅3D描述符(对照臂)")
add("L1_F5_A1", "L1", "F5", "A1", n_l1_pos, n_l1_neg, "★2D+3D(对照臂)")

# ---- 算法鲁棒性: L1 × F3 × A2/A3 ----
add("L1_F3_A2", "L1", "F3", "A2", n_l1_pos, n_l1_neg, "XGBoost 对照")
add("L1_F3_A3", "L1", "F3", "A3", n_l1_pos, n_l1_neg, "LogReg 对照(验证标签不变性)")

# ---- 标签敏感性: L2/L3/L4 × F3 × A1 ----
add("L2_F3_A1", "L2", "F3", "A1", n_l1_pos + len(unk_mat), n_l1_neg, "阳性放宽=Most+Less/Amb")
add("L3_F3_A1", "L3", "F3", "A1", np.nan, np.nan, "序数 2/1/0, 3类")
add("L4_F3_A1", "L4", "F3", "A1", np.nan, np.nan, "4类名义, 折叠 s=P(Most)+λP(Less)")

# ---- L4 的折叠权重扫描独立记录(不占模型矩阵行) ----
mat = pd.DataFrame(rows, columns=["model_id", "label_scheme", "feature_set",
                                  "algorithm", "n_train_pos", "n_train_neg", "notes"])
mat.to_csv(OUT / "S1_model_matrix.csv", index=False, encoding="utf-8-sig")
P(f"\n[1.4] 模型矩阵已写入 out/S1_model_matrix.csv  ({len(mat)} 行)")
P(mat.to_string(index=False))


# ======================================================================
# 4. 附加: L4 折叠权重扫描网格 (λ, μ)
# ======================================================================
grid = []
for lam in np.arange(0, 1.0001, 0.1):
    for mu in np.arange(0, 1.0001, 0.1):
        if mu <= lam + 1e-9:
            grid.append(dict(lam_less=round(float(lam), 2), mu_amb=round(float(mu), 2)))
grid_df = pd.DataFrame(grid)
grid_df.to_csv(OUT / "S1_fold_weight_grid.csv", index=False, encoding="utf-8-sig")
P(f"\n[1.5] L4 折叠权重网格 (0≤μ≤λ≤1): {len(grid_df)} 组合 -> out/S1_fold_weight_grid.csv")


# ======================================================================
# 5. 共享数据加载器
# ======================================================================
loader = r'''# -*- coding: utf-8 -*-
"""共享数据加载器 — 所有后续步骤 import 本模块, 保证 split / 路径一致(只读 D:\DILI)。"""
from pathlib import Path
import numpy as np
import pandas as pd

DILI = Path(r"D:\DILI")
DILI1 = Path(r"D:\DILI_1")
OUT = DILI1 / "out"
FIG = DILI1 / "figures"
CACHE = DILI1 / "cache"
for _d in (OUT, FIG, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
DESC_COLS = ["MolecularWeight", "LogP", "TPSA", "HBD", "HBA",
             "RotatableBonds", "AromaticRings"]
FP_COLS = [f"FP_{i}" for i in range(1024)]

_FDA = DILI / "data" / "Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx"
_DATASET = DILI / "data" / "DILI_ML_dataset_v5_final.xlsx"
_UNKRAW = DILI / "data" / "DILI_unknown_compounds_raw.xlsx"
_UNKMAT = DILI / "data" / "DILI_unknown_ML_matrix.xlsx"


def load_fda():
    """FDA DILIrank 2.0 原始 1336 药, 类别已规范化(修大小写坑)。"""
    fda = pd.read_excel(_FDA, header=1)
    c = fda["vDILI-Concern"].astype(str).str.strip()
    c = c.str.replace("vMOST", "vMost", regex=False).str.replace("Concern", "concern", regex=False)
    fda["DILI_category"] = c.map({
        "vMost-DILI-concern": "vMost",
        "vLess-DILI-concern": "vLess",
        "vNo-DILI-concern": "vNo",
        "Ambiguous-DILI-concern": "Ambiguous",
    })
    return fda


def load_dataset():
    """556 主数据集 (Most/No), 含 SMILES + 7desc + FP_0..1023。"""
    return pd.read_excel(_DATASET)


def load_fixed_split():
    """既有固定 split (test_size=0.2, seed=42, stratify) —— 所有模型必须复用。"""
    Xtr = pd.read_excel(DILI / "data" / "X_train.xlsx")
    Xte = pd.read_excel(DILI / "data" / "X_test.xlsx")
    ytr = pd.read_excel(DILI / "data" / "y_train.xlsx").values.ravel()
    yte = pd.read_excel(DILI / "data" / "y_test.xlsx").values.ravel()
    return Xtr, Xte, ytr, yte


def load_unknown():
    """vLess+Ambiguous 队列 (原始 + 已算特征矩阵)。"""
    raw = pd.read_excel(_UNKRAW)
    raw["DILI_category"] = raw["DILI_category"].astype(str).str.strip().map({
        "vLess-DILI-concern": "vLess",
        "Ambiguous-DILI-concern": "Ambiguous",
    })
    mat = pd.read_excel(_UNKMAT)
    return raw, mat


def feat_cols(feature_set, df=None):
    """按 feature_set ID 返回列名列表。F4/F5 需 3D 特征表(Step 2 产出)。"""
    fp = list(FP_COLS); desc = list(DESC_COLS)
    return {
        "F1": fp,
        "F2": desc,
        "F3": fp + desc,
    }.get(feature_set)


def make_models():
    """按算法 ID 返回新模型。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    def A1():
        return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                      random_state=SEED, n_jobs=-1)
    def A2():
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                             subsample=0.9, colsample_bytree=0.8,
                             scale_pos_weight=1.0, random_state=SEED,
                             eval_metric="logloss", n_jobs=-1)
    def A3():
        return LogisticRegression(class_weight="balanced", max_iter=5000,
                                  random_state=SEED)
    return {"A1": A1, "A2": A2, "A3": A3}
'''
(Path(r"D:\DILI_1\_scripts") / "dili_common.py").write_text(loader, encoding="utf-8")
P("\n[1.6] 共享加载器已写入 _scripts/dili_common.py")

(OUT / "S1_step1.log").write_text("\n".join(log), encoding="utf-8")
P("\nStep 1 完成。")
