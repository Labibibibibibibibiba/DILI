import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# =========================
# 文件路径
# =========================

data_path = r"D:\DILI\data\DILI_ML_dataset_v5_final.xlsx"


df = pd.read_excel(data_path)


print(df.head())
print(df.shape)


important_fp = [
    "FP_849",
    "FP_356",
    "FP_726",
    "FP_1",
    "FP_1019",
    "FP_926",
    "FP_981",
    "FP_367",
]

from collections import defaultdict

fp_occurrence = defaultdict(list)


for idx,row in df.iterrows():

    smiles=row["SMILES"]

    if pd.isna(smiles):
        continue

    mol=Chem.MolFromSmiles(smiles)

    if mol is None:
        continue


    bit_info={}


    fp=AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=1024,
        bitInfo=bit_info
    )


    for fp_name in important_fp:

        bit=int(fp_name.split("_")[1])


        if bit in bit_info:

            fp_occurrence[fp_name].append(
                {
                    "CompoundName":row["CompoundName"],
                    "DILI_label":row["DILI_label"],
                    "bitInfo":bit_info[bit]
                }
            )



for k,v in fp_occurrence.items():

    print("================")
    print(k)

    print("出现次数:",len(v))

    print("示例药物:")

    for item in v[:5]:
        print(
            item["CompoundName"],
            "DILI:",
            item["DILI_label"]
        )