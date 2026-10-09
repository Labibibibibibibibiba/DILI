import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score

from xgboost import XGBClassifier



# ==========================
# 1. 读取最终数据
# ==========================

df = pd.read_excel(
    "../data/DILI_ML_dataset_v5_final.xlsx"
)


print("数据规模:")
print(df.shape)



# ==========================
# 2. 构建X和y
# ==========================

X = df.drop(
    columns=[
        "CompoundName",
        "SMILES",
        "DILI_label"
    ]
)


y = df["DILI_label"]


print("\nX维度:")
print(X.shape)



print("\n标签分布:")
print(y.value_counts())



# ==========================
# 3. 五折交叉验证
# ==========================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)



# ==========================
# 4. XGBoost模型
# ==========================

model = XGBClassifier(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=3,

    subsample=0.8,

    colsample_bytree=0.8,

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1

)



# ==========================
# 5. Cross Validation
# ==========================

auc_scores = cross_val_score(

    model,

    X,

    y,

    cv=cv,

    scoring="roc_auc"

)



# ==========================
# 6. 输出结果
# ==========================


print("\n===================")

print("XGBoost")


print(
    "AUC per fold:"
)

print(
    auc_scores
)


print(
    "Mean AUC:",
    auc_scores.mean()
)


print(
    "STD:",
    auc_scores.std()
)


print("\n===================")

print("XGBoost CV完成")