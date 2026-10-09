import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem


# ==========================
# 文件
# ==========================

dataset_file = r"D:\DILI\data\DILI_ML_dataset_v5_final.xlsx"

output_file = r"D:\DILI\data\Acitretin_FP367_annotation.xlsx"



# ==========================
# 参数
# ==========================

compound_name = "Acitretin"

target_fp = 367



# ==========================
# 读取数据
# ==========================

df = pd.read_excel(dataset_file)


row = df[
    df["CompoundName"] == compound_name
]


if len(row)==0:
    raise Exception("没有找到Acitretin")



smiles = row.iloc[0]["SMILES"]


print("================")
print("SMILES:")
print(smiles)



mol = Chem.MolFromSmiles(smiles)


if mol is None:
    raise Exception("SMILES错误")



# ==========================
# Morgan bitInfo
# ==========================

bitInfo={}


fp = AllChem.GetMorganFingerprintAsBitVect(
    mol,
    radius=2,
    nBits=1024,
    bitInfo=bitInfo
)


print("================")
print("FP信息")

print(
    f"FP_{target_fp}:",
    bitInfo[target_fp]
)



# ==========================
# 获取原子范围
# ==========================

def get_environment_atoms(mol, atom_id, radius):

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


    return sorted(atoms), env



results=[]


bond_results=[]


for atom_id, radius in bitInfo[target_fp]:


    atoms, env = get_environment_atoms(
        mol,
        atom_id,
        radius
    )


    print("================")
    print(
        "center:",
        atom_id,
        "radius:",
        radius
    )


    print(
        "atoms:",
        atoms
    )



    # 原子信息

    for idx in atoms:

        atom = mol.GetAtomWithIdx(idx)


        results.append(
            {
                "Compound":compound_name,
                "Fingerprint":f"FP_{target_fp}",
                "Atom_ID":idx,
                "Element":atom.GetSymbol(),
                "Atomic_num":atom.GetAtomicNum(),
                "Degree":atom.GetDegree(),
                "Formal_charge":atom.GetFormalCharge()
            }
        )



    # 键信息

    for bond_id in env:

        bond=mol.GetBondWithIdx(bond_id)

        bond_results.append(
            {
                "Begin_atom":
                    bond.GetBeginAtomIdx(),

                "End_atom":
                    bond.GetEndAtomIdx(),

                "Bond_type":
                    str(bond.GetBondType())
            }
        )



# ==========================
# 子结构SMILES
# ==========================

submol = Chem.PathToSubmol(
    mol,
    list(env)
)


sub_smiles = Chem.MolToSmiles(
    submol
)


print("================")
print("Substructure:")
print(sub_smiles)



# ==========================
# 保存
# ==========================

atom_df=pd.DataFrame(results)

bond_df=pd.DataFrame(bond_results)


with pd.ExcelWriter(output_file) as writer:

    atom_df.to_excel(
        writer,
        sheet_name="Atoms",
        index=False
    )


    bond_df.to_excel(
        writer,
        sheet_name="Bonds",
        index=False
    )


    pd.DataFrame(
        {
            "Compound":[compound_name],
            "Fingerprint":[f"FP_{target_fp}"],
            "Substructure_SMILES":[sub_smiles],
            "Original_SMILES":[smiles]
        }
    ).to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )



print("================")
print("完成")

print(output_file)