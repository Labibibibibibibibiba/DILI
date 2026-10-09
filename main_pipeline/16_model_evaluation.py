import pandas as pd
import matplotlib.pyplot as plt

import joblib

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)



# ==========================
# 1. 读取测试集
# ==========================

X_test = pd.read_excel(
    "../data/X_test.xlsx"
)


y_test = pd.read_excel(
    "../data/y_test.xlsx"
)["DILI_label"]



# ==========================
# 2. 加载最佳RF模型
# ==========================

model = joblib.load(
    "../data/best_RF_model.pkl"
)



print("模型加载完成")



# ==========================
# 3. 获取预测概率
# ==========================

y_prob = model.predict_proba(
    X_test
)[:,1]



# ==========================
# ROC
# ==========================


auc_score = roc_auc_score(
    y_test,
    y_prob
)


fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)



plt.figure(figsize=(6,5))


plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC={auc_score:.3f})"
)


plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    "../data/ROC_curve.png",
    dpi=300
)


plt.close()



print("ROC AUC:")
print(auc_score)



# ==========================
# PR curve
# ==========================


precision, recall, pr_threshold = precision_recall_curve(

    y_test,

    y_prob

)


pr_auc = auc(
    recall,
    precision
)



plt.figure(figsize=(6,5))


plt.plot(
    recall,
    precision,
    label=f"PR-AUC={pr_auc:.3f}"
)



plt.xlabel(
    "Recall"
)


plt.ylabel(
    "Precision"
)


plt.title(
    "Precision-Recall Curve"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    "../data/PR_curve.png",
    dpi=300
)


plt.close()



print("PR-AUC:")
print(pr_auc)



# ==========================
# threshold=0.45
# ==========================


threshold = 0.45


y_pred = (
    y_prob >= threshold
).astype(int)



tn, fp, fn, tp = confusion_matrix(

    y_test,

    y_pred

).ravel()



print("\n===================")

print(
    "Threshold:",
    threshold
)



print(
    "Accuracy:",
    accuracy_score(
        y_test,
        y_pred
    )
)


print(
    "Precision:",
    precision_score(
        y_test,
        y_pred
    )
)


print(
    "Recall:",
    recall_score(
        y_test,
        y_pred
    )
)


print(
    "Specificity:",
    tn/(tn+fp)
)


print(
    "F1:",
    f1_score(
        y_test,
        y_pred
    )
)



print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)



print("\n===================")
print("模型评价完成")