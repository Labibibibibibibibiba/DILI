import pandas as pd


# ==========================
# 1. 读取数据
# ==========================

input_file = "../data/DILI_ML_dataset_v4_fingerprint.xlsx"

df = pd.read_excel(input_file)


print("原始数据:")
print(df.shape)


# ==========================
# 2. 删除无SMILES样本
# ==========================

df_ml = df.dropna(
    subset=["SMILES"]
)


print("\n删除无SMILES后:")
print(df_ml.shape)


# ==========================
# 3. 检查缺失值
# ==========================

missing = df_ml.isna().sum()

print("\n缺失值:")
print(
    missing[missing > 0]
)


# ==========================
# 4. 标签分布
# ==========================

print("\nDILI标签:")
print(
    df_ml["DILI_label"].value_counts()
)


# ==========================
# 5. 分离X和y
# ==========================

drop_columns = [
    "CompoundName",
    "SMILES",
    "DILI_label"
]


X = df_ml.drop(
    columns=drop_columns
)


y = df_ml["DILI_label"]


print("\nX维度:")
print(X.shape)


print("\ny维度:")
print(y.shape)



# ==========================
# 6. 保存
# ==========================


X.to_excel(
    "../data/X_features.xlsx",
    index=False
)


y.to_excel(
    "../data/y_label.xlsx",
    index=False
)


df_ml.to_excel(
    "../data/DILI_ML_ready_dataset.xlsx",
    index=False
)


print("\n全部完成")