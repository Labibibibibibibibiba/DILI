# -*- coding: utf-8 -*-
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
