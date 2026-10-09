import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)



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



print("数据读取完成")



# ==========================
# 2. 使用最佳RF参数
# ==========================

import joblib

model = joblib.load(
    "../data/best_RF_model.pkl"
)

model.fit(

    X_train,

    y_train

)



# ==========================
# 3. 获取预测概率
# ==========================


y_prob = model.predict_proba(

    X_test

)[:,1]



print("\nAUC:")

print(

    roc_auc_score(
        y_test,
        y_prob
    )

)



# ==========================
# 4. 不同threshold分析
# ==========================


thresholds = np.arange(

    0.2,

    0.81,

    0.05

)



results = []



for threshold in thresholds:


    y_pred = (

        y_prob >= threshold

    ).astype(int)



    tn, fp, fn, tp = confusion_matrix(

        y_test,

        y_pred

    ).ravel()



    results.append({


        "Threshold":
            round(threshold,2),


        "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),


        "Precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),


        "Recall":
            recall_score(
                y_test,
                y_pred
            ),


        "Specificity":
            tn/(tn+fp),


        "F1":
            f1_score(
                y_test,
                y_pred
            ),


        "TN":
            tn,


        "FP":
            fp,


        "FN":
            fn,


        "TP":
            tp

    })



result_df = pd.DataFrame(results)



print("\n===================")

print(result_df)



# ==========================
# 5. 保存结果
# ==========================


result_df.to_excel(

    "../data/RF_threshold_analysis.xlsx",

    index=False

)



print("\n结果已保存")
