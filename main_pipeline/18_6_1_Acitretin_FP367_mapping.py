import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


# ==========================
# 文件路径
# ==========================

dataset_file = r"D:\DILI\data\DILI_ML_dataset_v5_final.xlsx"


# ==========================
# 参数
# ==========================

target_compound = "Acitretin"
target_fp = 367


# ==========================
# 读取数据
# ==========================

df = pd.read_excel(dataset_file)


print("================")
print("数据:")
print(df.shape)


# 找Acitretin

row = df[
    df["CompoundName"] == target_compound
]


if len(row)==0:
    raise Exception("没有找到 Acitretin")


print("================")
print(row)



# ==========================
# 获取SMILES
# ==========================

# 根据你的数据列名调整
# 如果不是SMILES请告诉我

smiles = row.iloc[0]["SMILES"]


print("================")
print("SMILES:")
print(smiles)



mol = Chem.MolFromSmiles(smiles)


if mol is None:
    raise Exception("SMILES解析失败")



# ==========================
# Morgan bitInfo
# ==========================

bitInfo = {}


fp = AllChem.GetMorganFingerprintAsBitVect(
    mol,
    radius=2,
    nBits=1024,
    bitInfo=bitInfo
)



print("================")
print("bitInfo")


if target_fp not in bitInfo:

    raise Exception(
        f"Acitretin没有FP_{target_fp}"
    )


print(
    f"FP_{target_fp}:",
    bitInfo[target_fp]
)



# ==========================
# 提取原子范围
# ==========================

def get_atoms_from_env(mol, atom_id, radius):

    env = Chem.FindAtomEnvironmentOfRadiusN(
        mol,
        radius,
        atom_id
    )


    atoms=set()


    for bond_id in env:

        bond = mol.GetBondWithIdx(bond_id)

        atoms.add(
            bond.GetBeginAtomIdx()
        )

        atoms.add(
            bond.GetEndAtomIdx()
        )


    atoms.add(atom_id)


    return sorted(list(atoms))



# ==========================
# 输出
# ==========================


results=[]


for atom_id, radius in bitInfo[target_fp]:

    atoms=get_atoms_from_env(
        mol,
        atom_id,
        radius
    )


    results.append(
        {
            "center_atom":atom_id,
            "radius":radius,
            "atoms":atoms
        }
    )



result_df=pd.DataFrame(results)



print("================")
print("FP_367定位:")
print(result_df)



# ==========================
# 保存
# ==========================

output=r"D:\DILI\data\Acitretin_FP367_mapping.xlsx"


result_df.to_excel(
    output,
    index=False
)


print("================")
print("完成")
print(output)