import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ===============================
# 1. 文件路径
# ===============================

model_path = r"D:\DILI\data\best_RF_model.pkl"

feature_path = r"D:\DILI\data\X_features_final.xlsx"


# ===============================
# 2. 加载模型
# ===============================

model = joblib.load(model_path)

print("模型加载成功")



# ===============================
# 3. 加载feature
# ===============================

X = pd.read_excel(feature_path)

print("Feature shape:")
print(X.shape)

# ===============================
# 4. SHAP解释器
# ===============================

explainer = shap.TreeExplainer(model)

print("SHAP explainer建立完成")


# ===============================
# 5. 计算SHAP值
# ===============================

shap_values = explainer.shap_values(X)


print("SHAP计算完成")


# ===============================
# 6. 查看结构
# ===============================

print(type(shap_values))


# SHAP 0.51 二分类格式
# shape = (samples, features, classes)

if len(shap_values.shape) == 3:

    shap_value = shap_values[:,:,1]

else:

    shap_value = shap_values


print("用于分析的SHAP shape:")
print(shap_value.shape)


# ===============================
# 7. Feature重要性排序
# ===============================

importance = np.abs(shap_value).mean(axis=0)


feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "SHAP_importance": importance
})


feature_importance = (
    feature_importance
    .sort_values(
        by="SHAP_importance",
        ascending=False
    )
)


print("===================")
print("Top 20 important features")

print(
    feature_importance.head(20)
)


# 保存

feature_importance.to_excel(
    r"D:\DILI\result\SHAP_feature_importance.xlsx",
    index=False
)


# ===============================
# 8. SHAP summary plot
# ===============================

plt.figure(figsize=(10,8))


shap.summary_plot(
    shap_value,
    X,
    max_display=20,
    show=False
)


plt.tight_layout()


plt.savefig(
    r"D:\DILI\result\SHAP_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)


print("===================")
print("全部完成")