import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski


# ===============================
# 1. 读取数据
# ===============================

input_file = "../data/DILI_ML_dataset_v2.xlsx"

df = pd.read_excel(input_file)

print("原始数据:")
print(df.head())

print("数据规模:")
print(df.shape)


# ===============================
# 2. 定义特征计算函数
# ===============================

def calculate_descriptors(smiles):

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        descriptors = {

            "MolecularWeight":
                Descriptors.MolWt(mol),

            "LogP":
                Crippen.MolLogP(mol),

            "TPSA":
                Descriptors.TPSA(mol),

            "HBD":
                Lipinski.NumHDonors(mol),

            "HBA":
                Lipinski.NumHAcceptors(mol),

            "RotatableBonds":
                Lipinski.NumRotatableBonds(mol),

            "AromaticRings":
                Lipinski.NumAromaticRings(mol)
        }

        return descriptors


    except:

        return None



# ===============================
# 3. 批量计算
# ===============================

descriptor_list = []


for smiles in df["SMILES"]:

    if pd.isna(smiles):

        descriptor_list.append({})

    else:

        descriptor_list.append(
            calculate_descriptors(smiles)
        )


descriptor_df = pd.DataFrame(descriptor_list)


# 合并

result = pd.concat(
    [
        df,
        descriptor_df
    ],
    axis=1
)

print("描述符成功数量:")
print(descriptor_df["MolecularWeight"].notna().sum())


# ===============================
# 4. 保存
# ===============================

output_file = "../data/DILI_ML_dataset_v3_descriptor.xlsx"

result.to_excel(
    output_file,
    index=False
)


print("===================")
print("完成!")
print(result.shape)

print(result.head())