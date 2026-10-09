import pandas as pd


# ==========================
# 路径
# ==========================

input_file = r"D:\DILI\data\FP_structure_mapping_result.xlsx"

output_file = r"D:\DILI\data\representative_compounds.xlsx"


# ==========================
# 关注FP
# ==========================

target_fp = [
    "FP_367",
    "FP_849",
    "FP_981",
    "FP_726"
]


# ==========================
# 读取
# ==========================

df = pd.read_excel(input_file)


print("================")
print("原始:")
print(df.shape)



# ==========================
# 过滤目标FP
# ==========================

df = df[
    df["Fingerprint"].isin(target_fp)
]


# 去除重复结构定位

df_unique = df.drop_duplicates(
    subset=[
        "CompoundName",
        "Fingerprint"
    ]
)


print("目标FP:")
print(df_unique.shape)



# ==========================
# 优先DILI阳性
# ==========================

result=[]


for fp in target_fp:


    temp=df_unique[
        df_unique["Fingerprint"]==fp
    ]


    # DILI阳性优先

    temp=temp.sort_values(
        [
            "DILI_label",
            "CompoundName"
        ],
        ascending=[
            False,
            True
        ]
    )


    # 每个FP取前10

    temp=temp.head(10)


    result.append(temp)



final=pd.concat(
    result
)



print("================")
print(final[
    [
        "Fingerprint",
        "CompoundName",
        "DILI_label",
        "Substructure_SMILES"
    ]
])



final.to_excel(
    output_file,
    index=False
)


print("================")
print("完成")
print(output_file)