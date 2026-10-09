import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix
)

import seaborn as sns
import joblib


# ==========================
# paths
# ==========================

model_path = "../data/best_RF_model.pkl"

X_test_path = "../data/X_test.xlsx"

y_test_path = "../data/y_test.xlsx"



# ==========================
# load data
# ==========================

model = joblib.load(model_path)


X_test = pd.read_excel(
    X_test_path
)


y_test = pd.read_excel(
    y_test_path
)["DILI_label"]



# ==========================
# prediction
# ==========================


y_prob = model.predict_proba(
    X_test
)[:,1]


auc = roc_auc_score(
    y_test,
    y_prob
)


# threshold 0.45

threshold = 0.45


y_pred = (
    y_prob >= threshold
).astype(int)


cm = confusion_matrix(
    y_test,
    y_pred
)



print("AUC:",
      auc)

print(cm)



# ==========================
# Figure
# ==========================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10,4)
)



# --------------------------
# ROC
# --------------------------

fpr,tpr,_ = roc_curve(
    y_test,
    y_prob
)


axes[0].plot(
    fpr,
    tpr,
    linewidth=2
)


axes[0].plot(
    [0,1],
    [0,1],
    linestyle="--"
)


axes[0].set_xlabel(
    "False Positive Rate"
)

axes[0].set_ylabel(
    "True Positive Rate"
)


axes[0].set_title(
    "ROC curve"
)


axes[0].text(
    0.55,
    0.15,
    f"AUC = {auc:.3f}",
    fontsize=12
)



# --------------------------
# Confusion matrix
# --------------------------


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=axes[1]
)


axes[1].set_xlabel(
    "Predicted"
)


axes[1].set_ylabel(
    "Actual"
)


axes[1].set_title(
    "Confusion matrix\n(threshold=0.45)"
)



plt.tight_layout()



plt.savefig(
    "../result/Figure1D_test_performance.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()