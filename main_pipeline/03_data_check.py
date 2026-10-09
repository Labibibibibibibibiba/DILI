import pandas as pd


# 读取v2数据

data = pd.read_excel("../data/DILI_ML_dataset_v2.xlsx")


print("数据规模:")
print(data.shape)


print("\n列名:")
print(data.columns)


# 查看SMILES缺失情况

print("\nSMILES缺失数量:")
print(data["SMILES"].isna().sum())


# 查看标签分布

print("\nDILI标签分布:")
print(data["DILI_label"].value_counts())


# 查看前几条SMILES

print("\nSMILES示例:")
print(data[["CompoundName","SMILES"]].head())