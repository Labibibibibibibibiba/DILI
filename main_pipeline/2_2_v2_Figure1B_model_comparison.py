import matplotlib.pyplot as plt


# =========================
# Data
# =========================

models = [
    "Logistic\nRegression",
    "Random\nForest",
    "XGBoost",
    "Optimized\nRandom Forest"
]


auc = [
    0.7663,
    0.7990,
    0.7597,
    0.8106
]


# =========================
# Plot
# =========================

fig, ax = plt.subplots(
    figsize=(6,4)
)


bars = ax.bar(
    models,
    auc
)


# =========================
# Add values
# =========================

for bar, value in zip(bars, auc):

    ax.text(
        bar.get_x()+bar.get_width()/2,
        value+0.004,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )


# =========================
# Axis
# =========================

ax.set_ylabel(
    "Mean cross-validation AUC",
    fontsize=11
)


# focus on performance range

ax.set_ylim(
    0.70,
    0.85
)


ax.set_title(
    "Cross-validation performance of machine learning models",
    fontsize=13
)


# =========================
# Improve readability
# =========================

ax.tick_params(
    axis="x",
    labelsize=9
)


ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)


plt.tight_layout()


# =========================
# Save
# =========================

plt.savefig(
    "../result/Figure1B_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()