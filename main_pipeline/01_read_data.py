import pandas as pd


# 读取Excel文件
data = pd.read_excel("../data/DILI_ML_dataset_v1.xlsx")


# 查看前5行
print(data.head())


# 查看数据大小
print(data.shape)


# 查看列名
print(data.columns)
# 查看重复药物数量

print(data["CompoundName"].duplicated().sum())
print(data["DILI_label"].value_counts())