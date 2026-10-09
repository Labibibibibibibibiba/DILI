import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


# ==========================
# 1. 读取v3数据
# ==========================

input_file = "../data/DILI_ML_dataset_v3_descriptor.xlsx"

df = pd.read_excel(input_file)

print("数据规模:")
print(df.shape)


# ==========================
# 2. Morgan Fingerprint函数
# ==========================

def generate_fingerprint(smiles):

    try:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None


        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=1024
        )


        fp_list = list(fp)

        return fp_list


    except:

        return None



# ==========================
# 3. 批量生成fingerprint
# ==========================

fingerprint_list = []


for smiles in df["SMILES"]:

    if pd.isna(smiles):

        fingerprint_list.append([None]*1024)

    else:

        fp = generate_fingerprint(smiles)

        if fp is None:

            fingerprint_list.append([None]*1024)

        else:

            fingerprint_list.append(fp)



# ==========================
# 4. 转换为DataFrame
# ==========================

fp_columns = [
    f"FP_{i}"
    for i in range(1024)
]


fp_df = pd.DataFrame(
    fingerprint_list,
    columns=fp_columns
)



# ==========================
# 5. 合并
# ==========================

result = pd.concat(
    [
        df,
        fp_df
    ],
    axis=1
)



print("最终数据:")
print(result.shape)



# ==========================
# 6. 保存
# ==========================

output_file = "../data/DILI_ML_dataset_v4_fingerprint.xlsx"


result.to_excel(
    output_file,
    index=False
)


print("===================")
print("Fingerprint生成完成")