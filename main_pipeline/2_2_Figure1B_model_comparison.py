import matplotlib.pyplot as plt
import numpy as np


# =========================
# Data
# =========================

models = [
    "Logistic\nRegression",
    "Random\nForest",
    "XGBoost"
]


auc = [
    0.7663,
    0.7990,
    0.7597
]


# =========================
# Plot
# =========================

fig, ax = plt.subplots(
    figsize=(5,4)
)


bars = ax.bar(
    models,
    auc
)


# value labels

for bar, value in zip(bars, auc):

    ax.text(
        bar.get_x()+bar.get_width()/2,
        value+0.015,
        f"{value:.3f}",
        ha="center",
        fontsize=10
    )


# =========================
# Style
# =========================

ax.set_ylabel(
    "Mean cross-validation AUC",
    fontsize=11
)


ax.set_ylim(
    0,
    0.9
)


ax.set_title(
    "Comparison of machine learning models",
    fontsize=13
)


ax.tick_params(
    axis="x",
    labelsize=10
)


ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    "../result/Figure1B_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()