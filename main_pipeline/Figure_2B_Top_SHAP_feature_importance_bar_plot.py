# ============================================================
# Figure 2B
# Top SHAP feature importance bar plot
# ============================================================


import pandas as pd
import matplotlib.pyplot as plt


# ===============================
# 修改这里
# ===============================

input_file = r"D:\DILI\result\SHAP_feature_importance.xlsx"

output_file = r"D:\DILI\result\Figure2B_SHAP_feature_importance.png"


# ===============================
# 读取数据
# ===============================

df = pd.read_excel(input_file)


print("================")
print(df.head())
print(df.columns)
print("================")


# ===============================
# 检查列名
# ===============================

# 自动寻找feature列
feature_col = None
importance_col = None


for col in df.columns:

    if "Feature" in col:
        feature_col = col

    if "SHAP" in col or "importance" in col.lower():
        importance_col = col


print("Feature column:", feature_col)
print("Importance column:", importance_col)



# ===============================
# Top 10
# ===============================

top10 = (
    df.sort_values(
        by=importance_col,
        ascending=False
    )
    .head(10)
)


print(top10)



# ===============================
# 绘图
# ===============================

plt.figure(figsize=(7,5))


plt.barh(
    top10[feature_col][::-1],
    top10[importance_col][::-1]
)


plt.xlabel(
    "Mean absolute SHAP value"
)

plt.ylabel("")


plt.title(
    "Top contributing features identified by SHAP"
)


# 添加数值

for i, value in enumerate(top10[importance_col][::-1]):

    plt.text(
        value,
        i,
        f"{value:.4f}",
        va="center",
        fontsize=9
    )


plt.tight_layout()


plt.savefig(
    output_file,
    dpi=600,
    bbox_inches="tight"
)


plt.show()


print("================")
print("Figure 2B finished")
print(output_file)
print("================")