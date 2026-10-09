import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
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



print("训练集:")
print(X_train.shape)

print("测试集:")
print(X_test.shape)



# ==========================
# 2. 定义基础RF
# ==========================

rf = RandomForestClassifier(

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)



# ==========================
# 3. 参数搜索空间
# ==========================

param_grid = {


    "n_estimators":
        [100,300,500,800],


    "max_depth":
        [None,5,10,20],


    "min_samples_split":
        [2,5,10],


    "max_features":
        [
            "sqrt",
            "log2",
            0.5
        ]

}



# ==========================
# 4. Randomized Search
# ==========================

search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_grid,

    n_iter=30,

    scoring="roc_auc",

    cv=5,

    random_state=42,

    n_jobs=-1,

    verbose=1

)



print("\n开始搜索参数...")


search.fit(

    X_train,

    y_train

)



# ==========================
# 5. 输出最佳参数
# ==========================


print("\n===================")

print("最佳参数:")

print(
    search.best_params_
)


print(
    "最佳CV AUC:"
)

print(
    search.best_score_
)



# ==========================
# 6. 测试集评价
# ==========================


best_model = search.best_estimator_


y_prob = best_model.predict_proba(
    X_test
)[:,1]


y_pred = best_model.predict(
    X_test
)



print("\n===================")

print("Test Set Performance")


print(
    "AUC:",
    roc_auc_score(
        y_test,
        y_prob
    )
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

print("RF优化完成")

import joblib

joblib.dump(
    search.best_estimator_,
    "../data/best_RF_model.pkl"
)

print("最佳RF模型已保存")

from sklearn.model_selection import cross_val_score


print("\nOptimized RF cross-validation")

cv_scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)


print("AUC per fold:")
print(cv_scores)


print("Mean AUC:")
print(cv_scores.mean())


print("STD:")
print(cv_scores.std())