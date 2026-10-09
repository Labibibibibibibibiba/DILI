import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



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


print("\nX:")
print(X.shape)



# ==========================
# 3. 5-fold stratified CV
# ==========================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)



# ==========================
# 4. Logistic Regression
# ==========================


lr = LogisticRegression(

    max_iter=3000,

    class_weight="balanced",

    random_state=42

)



lr_auc = cross_val_score(

    lr,

    X,

    y,

    cv=cv,

    scoring="roc_auc"

)



print("\n===================")

print("Logistic Regression")

print(
    "AUC per fold:"
)

print(lr_auc)


print(
    "Mean AUC:",
    lr_auc.mean()
)


print(
    "STD:",
    lr_auc.std()
)



# ==========================
# 5. Random Forest
# ==========================


rf = RandomForestClassifier(

    n_estimators=500,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)



rf_auc = cross_val_score(

    rf,

    X,

    y,

    cv=cv,

    scoring="roc_auc"

)



print("\n===================")

print("Random Forest")


print(
    "AUC per fold:"
)

print(rf_auc)


print(
    "Mean AUC:",
    rf_auc.mean()
)


print(
    "STD:",
    rf_auc.std()
)



print("\n===================")
print("5-fold CV完成")