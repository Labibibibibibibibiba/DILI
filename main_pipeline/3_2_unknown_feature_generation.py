import pandas as pd
import numpy as np
import time

import pubchempy as pcp

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator



# ==========================
# Path
# ==========================

input_file = r"D:\DILI\data\DILI_unknown_compounds_raw.xlsx"

output_file = r"D:\DILI\data\DILI_unknown_ML_matrix.xlsx"

failed_file = r"D:\DILI\data\DILI_unknown_SMILES_failed.xlsx"



# ==========================
# Read compounds
# ==========================

df = pd.read_excel(
    input_file
)


print("====================")
print("Input compounds:")
print(df.shape)



# ==========================
# PubChem SMILES retrieval
# ==========================

def get_smiles(compound):

    try:

        results = pcp.get_compounds(
            compound,
            "name"
        )

        if len(results) > 0:

            return results[0].canonical_smiles

        else:

            return None


    except Exception:

        return None



smiles_list = []


failed = []


print("\nFetching PubChem SMILES...")



for i, name in enumerate(df["CompoundName"]):

    smiles = get_smiles(name)

    smiles_list.append(smiles)


    if smiles is None:

        failed.append(name)


    if (i+1) % 50 == 0:

        print(
            f"{i+1}/{len(df)} finished"
        )

    time.sleep(0.1)



df["SMILES"] = smiles_list



print("====================")

print(
    "SMILES missing:",
    df["SMILES"].isna().sum()
)



# save failed list

df[df["SMILES"].isna()].to_excel(
    failed_file,
    index=False
)



# ==========================
# Remove missing SMILES
# ==========================


df = df[
    df["SMILES"].notna()
].copy()



print(
    "After SMILES filtering:",
    df.shape
)



# ==========================
# RDKit molecule
# ==========================

df["Mol"] = df["SMILES"].apply(
    Chem.MolFromSmiles
)


df = df[
    df["Mol"].notna()
].copy()



# ==========================
# Molecular descriptors
# ==========================

def calc_descriptors(mol):

    return pd.Series({

        "MolecularWeight":
            Descriptors.MolWt(mol),

        "LogP":
            Crippen.MolLogP(mol),

        "TPSA":
            rdMolDescriptors.CalcTPSA(mol),

        "HBD":
            Lipinski.NumHDonors(mol),

        "HBA":
            Lipinski.NumHAcceptors(mol),

        "RotatableBonds":
            Lipinski.NumRotatableBonds(mol),

        "AromaticRings":
            rdMolDescriptors.CalcNumAromaticRings(mol)

    })



desc = df["Mol"].apply(
    calc_descriptors
)



# ==========================
# Morgan fingerprint
# ==========================


generator = GetMorganGenerator(
    radius=2,
    fpSize=1024
)



def get_fp(mol):

    fp = generator.GetFingerprint(mol)

    return list(fp)



fps = df["Mol"].apply(
    get_fp
)



fp_df = pd.DataFrame(
    fps.tolist(),
    columns=[
        f"FP_{i}"
        for i in range(1024)
    ]
)



# ==========================
# Combine
# ==========================


final_df = pd.concat(
    [
        df.drop(
            columns=["Mol"]
        ).reset_index(drop=True),

        desc.reset_index(drop=True),

        fp_df.reset_index(drop=True)

    ],
    axis=1
)



# ==========================
# Save
# ==========================


final_df.to_excel(
    output_file,
    index=False
)



print("====================")
print("Finished")

print(
    "Final shape:",
    final_df.shape
)


print(
    "Saved:",
    output_file
)