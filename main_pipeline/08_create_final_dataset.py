import pandas as pd


# ==========================
# 1. 读取v4数据
# ==========================

input_file = "../data/DILI_ML_dataset_v4_fingerprint.xlsx"

df = pd.read_excel(input_file)


print("原始数据:")
print(df.shape)


# ==========================
# 2. 删除无SMILES样本
# ==========================

df_final = df.dropna(
    subset=["SMILES"]
)


print("\n删除无SMILES后:")
print(df_final.shape)



# ==========================
# 3. 检查重复SMILES
# ==========================

duplicate_smiles = (
    df_final["SMILES"]
    .duplicated()
    .sum()
)


print("\n重复SMILES数量:")
print(duplicate_smiles)



# ==========================
# 4. 删除重复结构
# ==========================

df_final = (
    df_final
    .drop_duplicates(
        subset=["SMILES"],
        keep="first"
    )
)



print("\n删除重复结构后:")
print(df_final.shape)



# ==========================
# 5. 标签检查
# ==========================

print("\n最终标签分布:")
print(
    df_final["DILI_label"]
    .value_counts()
)



# ==========================
# 6. 保存
# ==========================

output_file = "../data/DILI_ML_dataset_v5_final.xlsx"


df_final.to_excel(
    output_file,
    index=False
)


print("\n===================")
print("最终数据集生成完成")
print(output_file)