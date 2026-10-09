import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# ==========================
# 路径
# ==========================

input_file = r"D:\DILI\data\FP_structure_mapping_result.xlsx"

output_file = r"D:\DILI\data\structure_enrichment_statistics.xlsx"

top_file = r"D:\DILI\data\top_risk_structures.xlsx"


# ==========================
# 参数
# ==========================

MIN_COUNT = 5



# ==========================
# 读取
# ==========================

df = pd.read_excel(input_file)


print("================")
print("原始:")
print(df.shape)



# 删除空结构

df = df.dropna(
    subset=[
        "Substructure_SMILES"
    ]
)



# 防止重复计数

df_unique = df.drop_duplicates(
    subset=[
        "CompoundName",
        "Fingerprint",
        "Substructure_SMILES"
    ]
)


print("去重:")
print(df_unique.shape)



# 总药物数

total_drugs = df_unique["CompoundName"].nunique()


print(
    "药物数量:",
    total_drugs
)



results=[]



# ==========================
# 遍历结构
# ==========================

for (fp, sub), group in df_unique.groupby(
    [
        "Fingerprint",
        "Substructure_SMILES"
    ]
):


    count=len(group)


    # 小样本过滤

    if count < MIN_COUNT:
        continue



    dili_pos=sum(
        group["DILI_label"]==1
    )

    dili_neg=sum(
        group["DILI_label"]==0
    )


    # 未出现该结构的药物

    other=df_unique[
        ~(
            (df_unique["Fingerprint"]==fp)
            &
            (df_unique["Substructure_SMILES"]==sub)
        )
    ]


    other_pos=sum(
        other["DILI_label"]==1
    )

    other_neg=sum(
        other["DILI_label"]==0
    )



    table=[
        [
            dili_pos,
            dili_neg
        ],
        [
            other_pos,
            other_neg
        ]
    ]



    # Fisher exact

    OR,p=fisher_exact(
        table
    )



    results.append(
        {

        "Fingerprint":fp,

        "Substructure_SMILES":sub,

        "Count":count,

        "DILI_positive":dili_pos,

        "DILI_negative":dili_neg,

        "DILI_rate":
            dili_pos/count,

        "OddsRatio":
            OR,

        "p_value":
            p

        }
    )



result=pd.DataFrame(results)



# ==========================
# FDR校正
# ==========================

reject, fdr, _, _ = multipletests(
    result["p_value"],
    method="fdr_bh"
)


result["FDR"]=fdr

result["Significant"]=reject



# ==========================
# 排序
# ==========================


result=result.sort_values(
    [
        "FDR",
        "OddsRatio"
    ],
    ascending=[
        True,
        False
    ]
)



print("================")
print("统计完成")

print(
    result.head(20)
)



# 保存

result.to_excel(
    output_file,
    index=False
)



# ==========================
# 风险结构
# ==========================


top=result[
    (result["Significant"])
    &
    (result["OddsRatio"]>1)
]


top=top.sort_values(
    "OddsRatio",
    ascending=False
)


top.to_excel(
    top_file,
    index=False
)



print("================")
print("完成")

print(
    "全部结果:",
    output_file
)

print(
    "风险结构:",
    top_file
)