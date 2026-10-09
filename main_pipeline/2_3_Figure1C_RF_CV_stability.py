import matplotlib.pyplot as plt
import numpy as np


# =========================
# CV AUC data
# =========================

auc_scores = np.array([
    0.79383117,
    0.84415584,
    0.78409091,
    0.77380952,
    0.85714286
])


mean_auc = auc_scores.mean()
std_auc = auc_scores.std()


folds = np.arange(1,6)



# =========================
# Figure
# =========================

plt.figure(
    figsize=(5,4)
)


# individual folds

plt.scatter(
    folds,
    auc_scores,
    s=80
)


# connect line

plt.plot(
    folds,
    auc_scores,
    linewidth=1.5
)



# mean line

plt.axhline(
    mean_auc,
    linestyle="--",
    linewidth=1.5,
    label=f"Mean AUC={mean_auc:.3f}"
)



# =========================
# Annotation
# =========================

plt.text(
    3.15,
    0.845,
    f"Mean AUC = {mean_auc:.3f}\nSD = {std_auc:.3f}",
    fontsize=11,
    ha="left",
    va="top"
)



# =========================
# Style
# =========================


plt.xlabel(
    "Cross-validation fold",
    fontsize=12
)


plt.ylabel(
    "AUC",
    fontsize=12
)


plt.title(
    "Optimized RF model stability",
    fontsize=13
)



plt.xticks(
    folds
)


plt.ylim(
    0.50,
    1.00
)


plt.grid(
    linestyle="--",
    alpha=0.3
)


plt.tight_layout()



# =========================
# Save
# =========================

plt.savefig(
    "../result/Figure1C_RF_CV_stability.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()