import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================
# 1. 读取训练测试数据
# ==========================

X_train = pd.read_excel(
    "../data/X_train.xlsx"
)

X_test = pd.read_excel(
    "../data/X_test.xlsx"
)


y_train = pd.read_excel(
    "../data/y_train.xlsx"
)["DILI_label"]


y_test = pd.read_excel(
    "../data/y_test.xlsx"
)["DILI_label"]



print("数据读取完成")

print("训练集:")
print(X_train.shape)

print("测试集:")
print(X_test.shape)



# ==========================
# 2. 定义评价函数
# ==========================

def evaluate_model(
        name,
        model,
        X_test,
        y_test
):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:,1]


    print("\n===================")
    print(name)


    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            y_pred
        )
    )


    print(
        "AUC:",
        roc_auc_score(
            y_test,
            y_prob
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



# ==========================
# 3. Logistic Regression
# ==========================

lr = LogisticRegression(

    max_iter=2000,

    class_weight="balanced",

    random_state=42

)


lr.fit(
    X_train,
    y_train
)


evaluate_model(
    "Logistic Regression",
    lr,
    X_test,
    y_test
)



# ==========================
# 4. Random Forest
# ==========================

rf = RandomForestClassifier(

    n_estimators=500,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)


rf.fit(
    X_train,
    y_train
)


evaluate_model(
    "Random Forest",
    rf,
    X_test,
    y_test
)



print("\n===================")
print("Baseline模型训练完成")