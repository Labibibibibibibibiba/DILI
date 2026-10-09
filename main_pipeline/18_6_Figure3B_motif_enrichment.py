import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==========================
# 文件路径
# ==========================

input_file = r"D:\DILI\data\structure_enrichment_statistics.xlsx"

output_file = r"D:\DILI\result\Figure3B_positive_motif_enrichment.png"


# ==========================
# 读取数据
# ==========================

df = pd.read_excel(input_file)


print(df)


# ==========================
# 筛选正相关motif
# OR > 1 且显著
# ==========================

df_pos = df[
    (df["OddsRatio"] > 1) &
    (df["Significant"] == True)
].copy()


# 排序

df_pos = df_pos.sort_values(
    "OddsRatio",
    ascending=True
)


# ==========================
# 名称转换
# ==========================

name_map = {

    "cc(c)C":
    "Methyl-substituted\naromatic fragment",

    "CNC(=O)CN":
    "Methylated amide\nfragment",

    "CCC(NC)C(N)=O":
    "Amino acid-like\nfragment",

    "CN":
    "Methylamine\nfragment"
}


df_pos["Motif"] = df_pos["Substructure_SMILES"].map(
    name_map
).fillna(
    df_pos["Substructure_SMILES"]
)


# ==========================
# 绘图
# ==========================

plt.figure(
    figsize=(6,4)
)


bars = plt.barh(
    df_pos["Motif"],
    df_pos["OddsRatio"]
)


# OR数字

for bar, or_value, fdr in zip(
    bars,
    df_pos["OddsRatio"],
    df_pos["FDR"]
):

    plt.text(
        bar.get_width()+0.1,
        bar.get_y()+bar.get_height()/2,
        f"OR={or_value:.2f}\nFDR={fdr:.3g}",
        va="center",
        fontsize=9
    )


# OR=1参考线

plt.axvline(
    1,
    linestyle="--"
)


plt.xlabel(
    "Odds ratio"
)

plt.ylabel(
    "Chemical motif"
)


plt.title(
    "Chemical motifs enriched in DILI-positive compounds"
)


plt.tight_layout()


plt.savefig(
    output_file,
    dpi=600,
    bbox_inches="tight"
)


plt.show()


print(
    "saved:",
    output_file
)