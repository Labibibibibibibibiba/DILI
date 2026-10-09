import pandas as pd

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score



# ==========================
# 1. 读取数据
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



print("原始feature数量:")
print(X_train.shape)



# ==========================
# 2. 删除低方差feature
# ==========================

selector_variance = VarianceThreshold(
    threshold=0
)


X_train_var = selector_variance.fit_transform(
    X_train
)


X_test_var = selector_variance.transform(
    X_test
)


print("\n删除低方差后:")
print(X_train_var.shape)



# ==========================
# 3. SelectKBest
# ==========================

k_values = [
    50,
    100,
    200
]


for k in k_values:


    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )


    X_train_k = selector.fit_transform(
        X_train_var,
        y_train
    )


    X_test_k = selector.transform(
        X_test_var
    )



    # Logistic Regression

    model = LogisticRegression(

        max_iter=3000,

        class_weight="balanced",

        random_state=42

    )


    model.fit(
        X_train_k,
        y_train
    )


    y_prob = model.predict_proba(
        X_test_k
    )[:,1]


    auc = roc_auc_score(
        y_test,
        y_prob
    )


    print(
        "\nFeature数量:",
        k
    )


    print(
        "AUC:",
        auc
    )



print("\n===================")
print("Feature selection完成")