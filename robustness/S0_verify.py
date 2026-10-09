"""
Step 0 —— 输入核对（严格只读 D:\\DILI，产物全部写入 D:\\DILI_1\\out）

用法:
  cd /d/DILI_1 && ./.venv/Scripts/python.exe _scripts/S0_verify.py *> out/S0.log 2>&1
"""
import io, json, os, sys, traceback

import pandas as pd

DATA = r"D:\DILI\data"
OUT = r"D:\DILI_1\out"
os.makedirs(OUT, exist_ok=True)

L = []
def p(*a):
    s = " ".join(str(x) for x in a)
    L.append(s)
    print(s)

report = {"errors": []}

def safe(name, fn):
    try:
        return fn()
    except Exception as e:
        p(f"[ERR] {name}: {type(e).__name__}: {e}")
        report["errors"].append(f"{name}: {type(e).__name__}: {e}")
        return None

p("=" * 72)
p("Step 0 输入核对  (只读 D:\\DILI)")
p("=" * 72)
p("python:", sys.version.split()[0], "|", sys.executable)
try:
    import numpy, sklearn, rdkit
    p("numpy:", numpy.__version__, "| sklearn:", sklearn.__version__, "| rdkit:", rdkit.__version__)
except Exception as e:
    p("deps warning:", e)

# ---------- 0.1 FDA DILIrank 2.0 原始四类分布 ----------
def fda():
    f = os.path.join(DATA, "Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx")
    df = pd.read_excel(f, header=1)
    p("\n[0.1] FDA raw shape:", df.shape)
    p("[0.1] columns:", list(df.columns))
    key = "vDILI-Concern"
    if key in df.columns:
        vc = df[key].value_counts(dropna=False)
        p("[0.1] vDILI-Concern 分布:")
        for k, v in vc.items():
            p(f"        {k!r:35s} {v}")
        report["fda_categories"] = {str(k): int(v) for k, v in vc.items()}
        report["fda_total"] = int(len(df))
    return df
safe("FDA raw", fda)

# ---------- 0.2 训练集标签 y_label_final ----------
def yfinal():
    df = pd.read_excel(os.path.join(DATA, "y_label_final.xlsx"))
    col = df.columns[0]
    vc = df[col].value_counts()
    p(f"\n[0.2] y_label_final shape={df.shape} col={col!r}")
    for k, v in vc.items():
        p(f"        label={k}  n={v}")
    report["y_final"] = {str(k): int(v) for k, v in vc.items()}
    report["n_final"] = int(len(df))
safe("y_label_final", yfinal)

# ---------- 0.3 既有 train/test split ----------
def split():
    ytr = pd.read_excel(os.path.join(DATA, "y_train.xlsx"))
    yte = pd.read_excel(os.path.join(DATA, "y_test.xlsx"))
    p(f"\n[0.3] y_train n={len(ytr)} -> {ytr.iloc[:,0].value_counts().to_dict()}")
    p(f"[0.3] y_test  n={len(yte)} -> {yte.iloc[:,0].value_counts().to_dict()}")
    report["train"] = {str(k): int(v) for k, v in ytr.iloc[:, 0].value_counts().items()}
    report["test"] = {str(k): int(v) for k, v in yte.iloc[:, 0].value_counts().items()}
    report["n_train"] = int(len(ytr))
    report["n_test"] = int(len(yte))
safe("train/test", split)

# ---------- 0.4 特征矩阵构成 ----------
def feats():
    X = pd.read_excel(os.path.join(DATA, "X_features_final.xlsx"))
    fp = [c for c in X.columns if str(c).startswith("FP_")]
    desc = [c for c in X.columns if not str(c).startswith("FP_")]
    p(f"\n[0.4] X_features_final shape={X.shape}")
    p(f"[0.4] FP 列 n={len(fp)}  (期望 1024)")
    p(f"[0.4] 非 FP 列 n={len(desc)} -> {desc}")
    p(f"[0.4] 期望总特征 = 1024 + 7 = 1031, 实际 = {X.shape[1]}")
    if len(fp) > 0:
        p(f"[0.4] FP 列样例: {fp[:3]} ... {fp[-2:]}")
    report["n_fp_cols"] = len(fp)
    report["desc_cols"] = desc
    report["n_features"] = int(X.shape[1])
safe("X_features_final", feats)

# ---------- 0.5 未知队列 (vLess + Ambiguous) ----------
def unknown():
    f1 = os.path.join(DATA, "DILI_unknown_compounds_raw.xlsx")
    if os.path.exists(f1):
        df = pd.read_excel(f1)
        p(f"\n[0.5] unknown raw shape={df.shape} cols={list(df.columns)}")
        if "DILI_category" in df.columns:
            vc = df["DILI_category"].value_counts()
            for k, v in vc.items():
                p(f"        {k!r:32s} {v}")
            report["unknown_raw"] = {str(k): int(v) for k, v in vc.items()}
    f2 = os.path.join(DATA, "DILI_unknown_prediction_rank.xlsx")
    if os.path.exists(f2):
        df2 = pd.read_excel(f2)
        p(f"[0.5] prediction_rank shape={df2.shape} cols={list(df2.columns)}")
        if "DILI_category" in df2.columns:
            vc = df2["DILI_category"].value_counts()
            for k, v in vc.items():
                p(f"        {k!r:32s} {v}")
            report["unknown_scored"] = {str(k): int(v) for k, v in vc.items()}
        if "DILI_risk_score" in df2.columns:
            s = df2["DILI_risk_score"]
            p(f"[0.5] risk_score: min={s.min():.4f} median={s.median():.4f} max={s.max():.4f}")
            report["risk_score_range"] = [float(s.min()), float(s.median()), float(s.max())]
safe("unknown cohort", unknown)

# ---------- 0.6 基线模型复现 ----------
def baseline():
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

    Xtr = pd.read_excel(os.path.join(DATA, "X_train.xlsx"))
    Xte = pd.read_excel(os.path.join(DATA, "X_test.xlsx"))
    ytr = pd.read_excel(os.path.join(DATA, "y_train.xlsx")).iloc[:, 0]
    yte = pd.read_excel(os.path.join(DATA, "y_test.xlsx")).iloc[:, 0]

    p(f"\n[0.6] X_train={Xtr.shape}  X_test={Xte.shape}")

    # 6a. 载入既有模型
    mp = os.path.join(DATA, "best_RF_model.pkl")
    if os.path.exists(mp):
        m = joblib.load(mp)
        prob = m.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, prob)
        p(f"[0.6a] 既有 best_RF_model.pkl  测试集 AUC = {auc:.4f}")
        for t in (0.45, 0.5):
            pred = (prob >= t).astype(int)
            p(f"        threshold={t}: F1={f1_score(yte, pred):.4f}  CM={confusion_matrix(yte, pred).tolist()}")
        report["baseline_pkl_auc"] = float(auc)

    # 6b. 重新拟合一个同规格 RF（无调参），看 CV
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                               random_state=42, n_jobs=-1)
    cv = cross_val_score(rf, Xtr, ytr, cv=5, scoring="roc_auc", n_jobs=-1)
    p(f"[0.6b] 基线 RF(500) 5-fold CV AUC = {cv.mean():.4f} ± {cv.std():.4f}")
    p(f"        每折: {[round(x,4) for x in cv]}")
    report["baseline_cv_auc_mean"] = float(cv.mean())
    report["baseline_cv_auc_std"] = float(cv.std())
safe("baseline reproduce", baseline)

# ---------- 输出 ----------
p("\n" + "=" * 72)
p("核对完成。errors:", len(report["errors"]))
for e in report["errors"]:
    p("  -", e)

with io.open(os.path.join(OUT, "S0_inventory.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
with io.open(os.path.join(OUT, "S0_baseline_repro.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("\nWROTE out/S0_inventory.txt and out/S0_baseline_repro.json")
