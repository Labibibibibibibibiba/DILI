import pandas as pd


# ==========================
# 文件路径
# ==========================

input_file = r"D:\DILI\data\FP_structure_mapping_result.xlsx"


output_file = r"D:\DILI\data\FP_structure_enrichment.xlsx"


top_output = r"D:\DILI\data\FP_top_structures.xlsx"



# ==========================
# 读取18.3结果
# ==========================

df = pd.read_excel(input_file)


print("================")
print("输入数据:")
print(df.shape)



# ==========================
# 删除空结构
# ==========================

df = df.dropna(
    subset=["Substructure_SMILES"]
)


print("有效结构:")
print(df.shape)



# ==========================
# 去除同一药物同一结构重复
# 防止大分子重复计数
# ==========================

df_unique = df.drop_duplicates(
    subset=[
        "CompoundName",
        "Fingerprint",
        "Substructure_SMILES"
    ]
)


print("去重后:")
print(df_unique.shape)



# ==========================
# 结构富集统计
# ==========================


results=[]


for (fp, sub), group in df_unique.groupby(
    [
        "Fingerprint",
        "Substructure_SMILES"
    ]
):


    total=len(group)


    dili_positive=sum(
        group["DILI_label"]==1
    )

    dili_negative=sum(
        group["DILI_label"]==0
    )


    rate=dili_positive/total


    results.append(
        {
            "Fingerprint":fp,

            "Substructure_SMILES":sub,

            "Total":total,

            "DILI_positive":dili_positive,

            "DILI_negative":dili_negative,

            "DILI_rate":rate
        }
    )



result_df=pd.DataFrame(results)



# ==========================
# 排序
# ==========================

result_df=result_df.sort_values(
    [
        "Fingerprint",
        "DILI_rate",
        "Total"
    ],
    ascending=[
        True,
        False,
        False
    ]
)



print("================")
print("结构统计完成")
print(result_df.head(20))



result_df.to_excel(
    output_file,
    index=False
)



# ==========================
# 每个FP top结构
# ==========================


top_df = (
    result_df
    .groupby("Fingerprint")
    .head(10)
)



top_df.to_excel(
    top_output,
    index=False
)



print("================")
print("完成")
print(output_file)
print(top_output)