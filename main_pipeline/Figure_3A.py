import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ================================
# 修改路径
# ================================

file = r"D:\DILI\data\FP_enrichment_results.xlsx"

output = r"D:\DILI\result\Figure3A_FP_enrichment.png"



# ================================
# 读取
# ================================

df = pd.read_excel(file)


print(df.head())



# ================================
# 选择显著feature
# ================================


features = [
    "FP_356",
    "FP_367",
    "FP_849",
    "FP_726",
    "FP_981"
]


plot_df = df[
    df["Feature"].isin(features)
].copy()



# 排序

plot_df = plot_df.sort_values(
    "OddsRatio"
)



# ================================
# 绘图
# ================================


plt.figure(figsize=(8,5))


plt.barh(
    plot_df["Feature"],
    plot_df["OddsRatio"]
)



# OR标注

for i,v in enumerate(plot_df["OddsRatio"]):

    plt.text(
        v+0.05,
        i,
        f"{v:.2f}",
        va="center"
    )



plt.axvline(
    1,
    linestyle="--"
)



plt.xlabel(
    "Odds ratio"
)


plt.ylabel(
    "Fingerprint"
)


plt.title(
    "Fingerprint enrichment associated with DILI classification"
)


plt.tight_layout()


plt.savefig(
    output,
    dpi=600,
    bbox_inches="tight"
)


plt.show()


print("done")