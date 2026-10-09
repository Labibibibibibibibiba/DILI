#!/usr/bin/env python3
"""
Figure 1 — Development and validation of an interpretable RF-based DILI prediction model.

布局：双栏宽 183 mm，上下结构
  A（顶部整行）：workflow 横向流程（数据→特征→划分→建模→验证）
  B（底部左）：DILIrank 2.0 四分类组成
  C（底部中）：LR / RF / XGB 三模型 5-fold CV AUC 对比（箱线图）
  D（底部右）：RF 测试性能（ROC 曲线 + 混淆矩阵 @ threshold 0.45）
"""

import sys
sys.path.insert(0, r"D:/abraca")
import plot_utils as pu
pu.apply_paper_style()

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import joblib
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# =============================================================================
# 数据加载 + 重算（v2：阈值在训练集 OOF 上选取，测试集只评估一次）
# =============================================================================
M = json.load(open(r"D:/DILI/data/model_metrics_v2.json"))
TAU = M["threshold_selection"]["tau_star_f1"]

model = joblib.load(r"D:/DILI/data/best_RF_model_v2.pkl")
X_test = pd.read_excel(r"D:/DILI/data/X_test.xlsx")
y_test = pd.read_excel(r"D:/DILI/data/y_test.xlsx")["DILI_label"]
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)
y_pred = (y_prob >= TAU).astype(int)
cm = confusion_matrix(y_test, y_pred)
print("Figure1: AUC=%.4f  tau*=%.2f  CM=%s" % (auc, TAU, cm.tolist()))

# 三模型 5-fold CV AUC（口径：训练集 444，无测试集参与）
cv_models = [
    ("LR",  M["cv"]["LR__train444"]["folds"], pu.MODEL_COLORS["Logistic Regression"]),
    ("RF",  M["cv"]["RF_baseline__train444"]["folds"], pu.MODEL_COLORS["Random Forest"]),
    ("XGB", M["cv"]["XGB__train444"]["folds"], pu.MODEL_COLORS["XGBoost"]),
]

# DILIrank 2.0 四分类数量（FDA 官方）
dili_classes = [
    ("vMost", 217, pu.DILI_CLASS_COLORS["vMost-DILI-concern"]),
    ("vLess", 351, pu.DILI_CLASS_COLORS["vLess-DILI-concern"]),
    ("vNo",   414, pu.DILI_CLASS_COLORS["vNo-DILI-concern"]),
    ("Ambig", 354, pu.DILI_CLASS_COLORS["Ambiguous-DILI-concern"]),
]

# =============================================================================
# 画布：双栏宽，上下结构
# =============================================================================
fig = plt.figure(figsize=pu.figsize(pu.DOUBLE_COL_MM, height_mm=108))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.3],
                      width_ratios=[1.05, 1.0, 0.9, 0.9],
                      hspace=0.55, wspace=0.5)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])
axD1 = fig.add_subplot(gs[1, 2])    # ROC
axD2 = fig.add_subplot(gs[1, 3])    # 混淆矩阵

# =============================================================================
# A：workflow（横向流程）
# =============================================================================
axA.axis("off")
axA.set_xlim(0, 1)
axA.set_ylim(0, 1)
steps = [
    ("DILIrank 2.0", "Most 217\nNo 414", False),
    ("Features", "RDKit + Morgan\n1031 features", False),
    ("Split", "444 train\n112 test", False),
    ("Model", "LR / RF / XGB\n\u2192 RF", True),
    ("Validation", "AUROC {:.3f}\n\u2192 Fig 2 / 3".format(auc), False),
]
x0, w, gap = 0.005, 0.185, 0.017
for i, (title, sub, hl) in enumerate(steps):
    x = x0 + i * (w + gap)
    box = FancyBboxPatch((x, 0.16), w, 0.68, boxstyle="round,pad=0.012",
                         linewidth=0.7,
                         edgecolor="#888780",
                         facecolor="#E6F1FB" if hl else "#FFFFFF")
    axA.add_patch(box)
    axA.text(x + w / 2, 0.65, title, ha="center", va="center",
             fontsize=7.5, fontweight="bold" if hl else "normal")
    axA.text(x + w / 2, 0.38, sub, ha="center", va="center", fontsize=6.5)
    if i < 4:
        axA.annotate("", xy=(x + w + gap - 0.003, 0.5), xytext=(x + w + 0.003, 0.5),
                     arrowprops=dict(arrowstyle="-|>", color="#888780", lw=0.8))
pu.label_panel(axA, "A")

# =============================================================================
# B：DILIrank 2.0 四分类组成（柱状图）
# =============================================================================
labels = [c[0] for c in dili_classes]
values = [c[1] for c in dili_classes]
colors = [c[2] for c in dili_classes]
bars = axB.bar(labels, values, color=colors, edgecolor="black",
               linewidth=0.5, width=0.62)
for bar, v in zip(bars, values):
    axB.text(bar.get_x() + bar.get_width() / 2, v + 8, str(v),
             ha="center", va="bottom", fontsize=7)
# 训练用的 Most/No 加粗边框
for i, name in enumerate(labels):
    if name in ("vMost", "vNo"):
        bars[i].set_linewidth(1.5)
        bars[i].set_edgecolor("black")
axB.set_ylabel("Number of compounds")
axB.set_ylim(0, 460)
axB.tick_params(axis="x", labelsize=7)
pu.label_panel(axB, "B", y=1.2)

# =============================================================================
# C：三模型 5-fold CV AUC（箱线图 + per-fold 散点）
# =============================================================================
model_names = [m[0] for m in cv_models]
fold_data = [m[1] for m in cv_models]
box_colors = [m[2] for m in cv_models]
bp = axC.boxplot(fold_data, tick_labels=model_names, patch_artist=True,
                 widths=0.5, showfliers=False,
                 medianprops=dict(color="black", linewidth=1.0))
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)
rng = np.random.default_rng(7)
for i, d in enumerate(fold_data):
    jitter = rng.normal(i + 1, 0.04, len(d))
    axC.scatter(jitter, d, color="black", s=5, zorder=3, alpha=0.7)
axC.axhline(0.5, linestyle="--", color="#B4B2A9", linewidth=0.6, zorder=0)
axC.set_ylabel("5-fold CV AUC")
axC.set_ylim(0.5, 0.92)
axC.text(2, 0.90, "selected", ha="center", fontsize=6.5,
         color=pu.OKABE_ITO["blue"], fontweight="bold")
pu.label_panel(axC, "C", y=1.2)

# =============================================================================
# D：ROC + 混淆矩阵
# =============================================================================
# ROC
axD1.plot(fpr, tpr, color=pu.OKABE_ITO["blue"], linewidth=1.5)
axD1.plot([0, 1], [0, 1], linestyle="--", color="#B4B2A9", linewidth=0.8)
axD1.set_xlabel("False positive rate")
axD1.set_ylabel("True positive rate")
axD1.set_xlim(0, 1)
axD1.set_ylim(0, 1)
axD1.text(0.56, 0.12, f"AUC = {auc:.3f}", fontsize=7)
# 混淆矩阵
im = axD2.imshow(cm, cmap="Blues", vmin=0, vmax=60)
axD2.set_xticks([0, 1]); axD2.set_xticklabels(["No-DILI", "Most-DILI"], fontsize=6)
axD2.set_yticks([0, 1]); axD2.set_yticklabels(["No-DILI", "Most-DILI"], fontsize=6)
axD2.set_xlabel("Predicted", fontsize=6.5)
axD2.set_ylabel("Actual", fontsize=6.5)
axD2.tick_params(axis="x", rotation=0)
for i in range(2):
    for j in range(2):
        axD2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                  color="white" if cm[i, j] > 30 else "black")
pu.label_panel(axD1, "D", y=1.2)

# =============================================================================
# 导出
# =============================================================================
pu.save_figure(fig, r"D:/DILI/result/new_figure/Figure1",
               formats=("pdf", "png"), dpi=600, tight=True)
plt.close(fig)
print("Figure1 完成")
