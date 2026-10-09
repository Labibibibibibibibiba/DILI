import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================
# 1. 读取最终数据
# ==========================

input_file = "../data/DILI_ML_dataset_v5_final.xlsx"

df = pd.read_excel(input_file)


print("最终数据规模:")
print(df.shape)



# ==========================
# 2. 提取X和y
# ==========================

drop_columns = [
    "CompoundName",
    "SMILES",
    "DILI_label"
]


X = df.drop(
    columns=drop_columns
)


y = df["DILI_label"]



print("\nX维度:")
print(X.shape)


print("\ny维度:")
print(y.shape)



# ==========================
# 3. 标签检查
# ==========================

print("\n标签分布:")
print(
    y.value_counts()
)



# ==========================
# 4. 保存最终X/y
# ==========================

X.to_excel(
    "../data/X_features_final.xlsx",
    index=False
)


y.to_excel(
    "../data/y_label_final.xlsx",
    index=False
)



# ==========================
# 5. Train/Test split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



print("\n训练集:")
print(X_train.shape)

print(y_train.value_counts())


print("\n测试集:")
print(X_test.shape)

print(y_test.value_counts())



# ==========================
# 6. 保存训练测试数据
# ==========================


X_train.to_excel(
    "../data/X_train.xlsx",
    index=False
)


X_test.to_excel(
    "../data/X_test.xlsx",
    index=False
)


y_train.to_excel(
    "../data/y_train.xlsx",
    index=False
)


y_test.to_excel(
    "../data/y_test.xlsx",
    index=False
)



print("\n===================")
print("最终机器学习数据准备完成")