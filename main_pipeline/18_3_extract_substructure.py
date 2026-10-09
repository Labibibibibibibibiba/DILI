# ==========================================
# 18_3_FP_structure_mapping.py
# Morgan fingerprint -> chemical substructure
# ==========================================

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem



# ==============================
# 文件路径
# ==============================

input_file = r"D:\DILI\data\DILI_ML_dataset_v5_final.xlsx"

output_file = r"D:\DILI\data\FP_structure_mapping_result.xlsx"


# ==============================
# 需要解释的fingerprint
# 来自SHAP +统计分析
# ==============================

target_fps = [
    "FP_356",
    "FP_367",
    "FP_849",
    "FP_726",
    "FP_1019",
    "FP_926",
    "FP_1",
    "FP_981"
]


target_bits = [
    int(x.split("_")[1])
    for x in target_fps
]


# ==============================
# 读取数据
# ==============================

df = pd.read_excel(input_file)


print("================")
print("数据:")
print(df.shape)


# ==============================
# 结果保存
# ==============================

results = []


debug_count = 0


# ==============================
# 主循环
# ==============================

for idx, row in df.iterrows():

    name = row["CompoundName"]
    smiles = row["SMILES"]


    if pd.isna(smiles):
        continue


    mol = Chem.MolFromSmiles(smiles)


    if mol is None:
        continue


    # --------------------------
    # 关键：
    # 必须和模型一致
    # --------------------------

    bit_info = {}

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=1024,
        bitInfo=bit_info
    )


    # --------------------------
    # 检查目标FP
    # --------------------------

    for bit, locations in bit_info.items():


        if bit not in target_bits:
            continue


        if debug_count < 20:

            print(
                "FOUND:",
                name,
                f"FP_{bit}",
                locations
            )

            debug_count += 1


        # locations:
        # ((atom_id, radius), ...)

        for atom_id, radius in locations:


            try:


                # =========================
                # 获取原子环境
                # RDKit新版参数顺序：
                # mol, radius, atom
                # =========================

                env = Chem.FindAtomEnvironmentOfRadiusN(
                    mol,
                    radius,
                    atom_id
                )


                if len(env)==0:
                    continue


                # =========================
                # 根据bond环境生成子结构
                # =========================

                submol = Chem.PathToSubmol(
                    mol,
                    env
                )


                sub_smiles = Chem.MolToSmiles(
                    submol
                )


                results.append(
                    {

                        "CompoundName": name,

                        "DILI_label": row["DILI_label"],

                        "Fingerprint":
                            f"FP_{bit}",

                        "Atom_ID":
                            atom_id,

                        "Radius":
                            radius,

                        "Substructure_SMILES":
                            sub_smiles
                    }
                )


            except Exception as e:

                print(
                    "Error:",
                    name,
                    f"FP_{bit}",
                    e
                )



# ==============================
# 保存
# ==============================

result_df = pd.DataFrame(results)


print("================")
print("完成")
print("结果数量:")
print(result_df.shape)

print("================")
print(result_df.head())


result_df.to_excel(
    output_file,
    index=False
)


print("================")
print("保存完成:")
print(output_file)