import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import fisher_exact


# =====================
# 数据
# =====================

data_path = r"D:\DILI\data\DILI_ML_dataset_v5_final.xlsx"

df = pd.read_excel(data_path)


important_fp = [
    "FP_849",
    "FP_356",
    "FP_726",
    "FP_1",
    "FP_1019",
    "FP_926",
    "FP_981",
    "FP_367"
]


# =====================
# 统计
# =====================

results=[]


for fp_name in important_fp:

    bit=int(fp_name.split("_")[1])


    has_fp=[]
    labels=[]


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


        if bit in bit_info:
            has_fp.append(1)
        else:
            has_fp.append(0)


        labels.append(row["DILI_label"])



    temp=pd.DataFrame({
        "FP":has_fp,
        "label":labels
    })


    a=sum((temp.FP==1)&(temp.label==1))
    b=sum((temp.FP==1)&(temp.label==0))
    c=sum((temp.FP==0)&(temp.label==1))
    d=sum((temp.FP==0)&(temp.label==0))


    table=[
        [a,b],
        [c,d]
    ]


    odds,p=fisher_exact(table)


    results.append(
        {
            "Feature":fp_name,
            "FP_DILI_positive":a,
            "FP_DILI_negative":b,
            "NoFP_DILI_positive":c,
            "NoFP_DILI_negative":d,
            "OddsRatio":odds,
            "p_value":p
        }
    )



result_df=pd.DataFrame(results)


result_df=result_df.sort_values(
    "OddsRatio",
    ascending=False
)


print(result_df)


result_df.to_excel(
    r"D:\DILI\data\FP_enrichment_results.xlsx",
    index=False
)

print("完成")