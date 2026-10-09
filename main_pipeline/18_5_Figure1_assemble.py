import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns


# ==============================
# 需要修改的路径
# ==============================

roc_path = r"D:\DILI\data\ROC_curve.png"

pr_path = r"D:\DILI\data\PR_curve.png"

threshold_excel = r"D:\DILI\data\RF_threshold_analysis.xlsx"

output_path = r"D:\DILI\result\Figure1_Model_Performance.png"


# ==============================
# 读取threshold数据
# ==============================

df = pd.read_excel(threshold_excel)


print(df.head())
print(df.columns)


# ==============================
# 创建画布
# ==============================

fig = plt.figure(figsize=(14,12))


# ==============================
# A. ROC
# ==============================

ax1 = plt.subplot(2,2,1)

roc_img = mpimg.imread(roc_path)

ax1.imshow(roc_img)

ax1.axis("off")

ax1.text(
    -0.05,
    1.05,
    "A",
    transform=ax1.transAxes,
    fontsize=20,
    fontweight="bold"
)



# ==============================
# B. PR curve
# ==============================

ax2 = plt.subplot(2,2,2)

pr_img = mpimg.imread(pr_path)

ax2.imshow(pr_img)

ax2.axis("off")

ax2.text(
    -0.05,
    1.05,
    "B",
    transform=ax2.transAxes,
    fontsize=20,
    fontweight="bold"
)



# ==============================
# C. Threshold analysis
# ==============================

ax3 = plt.subplot(2,2,3)


ax3.plot(
    df["Threshold"],
    df["Recall"],
    marker="o",
    label="Recall"
)


ax3.plot(
    df["Threshold"],
    df["Precision"],
    marker="o",
    label="Precision"
)


ax3.plot(
    df["Threshold"],
    df["F1"],
    marker="o",
    label="F1-score"
)



# 标记最佳threshold

best_threshold = 0.45


best_row = df[
    df["Threshold"] == best_threshold
]


ax3.scatter(
    best_threshold,
    best_row["F1"].values[0],
    s=150
)


ax3.annotate(
    "Optimal threshold=0.45",
    xy=(
        best_threshold,
        best_row["F1"].values[0]
    ),
    xytext=(
        0.55,
        0.8
    ),
    arrowprops=dict(
        arrowstyle="->"
    ),
    fontsize=11
)



ax3.set_xlabel(
    "Threshold",
    fontsize=12
)


ax3.set_ylabel(
    "Score",
    fontsize=12
)


ax3.set_ylim(0,1)


ax3.legend()


ax3.grid(alpha=0.3)


ax3.text(
    -0.15,
    1.05,
    "C",
    transform=ax3.transAxes,
    fontsize=20,
    fontweight="bold"
)



# ==============================
# D. Confusion matrix
# ==============================


ax4 = plt.subplot(2,2,4)


cm = np.array(
    [
        [48,23],
        [4,37]
    ]
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=ax4
)



ax4.set_xlabel(
    "Predicted",
    fontsize=12
)


ax4.set_ylabel(
    "Actual",
    fontsize=12
)


ax4.set_xticklabels(
    [
        "Negative",
        "Positive"
    ]
)


ax4.set_yticklabels(
    [
        "Negative",
        "Positive"
    ],
    rotation=0
)



ax4.text(
    -0.15,
    1.05,
    "D",
    transform=ax4.transAxes,
    fontsize=20,
    fontweight="bold"
)



# ==============================
# 总标题
# ==============================


plt.suptitle(
    "Performance evaluation of the RF-based DILI risk prediction model",
    fontsize=18,
    y=0.98
)


plt.tight_layout()


# ==============================
# 保存
# ==============================


plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print("====================")
print("Figure 1 finished")
print(output_path)
print("====================")