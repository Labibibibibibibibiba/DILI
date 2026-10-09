import pandas as pd


data = pd.read_excel("../data/DILI_ML_dataset_v2.xlsx")


missing = data[data["SMILES"].isna()]


print("缺失数量:")
print(len(missing))


print("\n缺失药物列表:")
print(missing["CompoundName"].to_string(index=False))