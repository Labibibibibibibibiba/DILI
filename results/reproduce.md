# 复现指南 — DILI_1 多模型比较

> 从零复现全流程。所有输入只读 `D:\DILI`，所有产物写入 `D:\DILI_1`。
> 最后更新：2026-10-04

---

## 0. 前置条件

| 项 | 要求 |
|---|---|
| Python | 3.13.14（本机 `C:\Users\sumsc\.workbuddy\binaries\python\versions\3.13.12\python.exe`） |
| 输入数据 | `D:\DILI\data\` 下的 8 个文件（见交接文件 §2） |
| 随机种子 | 全部固定 `SEED = 42` |
| 磁盘 | `D:\DILI_1` 可写；**`D:\DILI` 只读，严禁写入** |

⚠️ **不要使用 `D:\DILI\code\.venv`** —— 它已损坏（base interpreter 指向不存在的 `D:\py\python.exe`）。

---

## 1. 建环境

```bash
cd /d/DILI_1
"C:/Users/sumsc/.workbuddy/binaries/python/versions/3.13.12/python.exe" -m venv .venv
./.venv/Scripts/python.exe -m pip install --retries 8 --timeout 180 \
    pandas openpyxl scikit-learn rdkit joblib scipy numpy matplotlib xgboost
```

> 网络受限提示：本机 pip 走默认源可用；若镜像 502，**逐个包安装**，避免单包失败导致整批事务回滚。

验证：

```bash
./.venv/Scripts/python.exe -c "import pandas,sklearn,rdkit,xgboost,matplotlib;print('OK')"
```

依赖版本见 `out/requirements_frozen.txt`。

---

## 2. 按序执行

```bash
cd /d/DILI_1
PY=./.venv/Scripts/python.exe

$PY _scripts/S0_verify.py              *> out/S0.log 2>&1   # Step 0 输入核对 + 基线复现 (G1)
$PY _scripts/S1_model_matrix.py        *> out/S1_run.log 2>&1  # Step 1 模型矩阵落表
$PY _scripts/S2_features3d.py          *> out/S2_run.log 2>&1  # Step 2 3D 特征 (首次, MMFF 可能失败)
$PY _scripts/S2b_features3d_fix.py     *> out/S2b_run.log 2>&1 # Step 2b 3D 补救 (MMFF→UFF→ETKDG, ~29 min)
$PY _scripts/S3_label_sensitivity.py   *> out/S3_run.log 2>&1  # Step 3 标签敏感性
$PY _scripts/S4_descriptor_compare.py  *> out/S4_run.log 2>&1  # Step 4 描述符空间 (G3)
$PY _scripts/S5_cross_model_sets.py    *> out/S5_run.log 2>&1  # Step 5 跨模型集合 (G4, 核心)
$PY _scripts/S6_wetlab_candidates.py   *> out/S6_run.log 2>&1  # Step 6 PHH 候选
$PY _scripts/S7_figures.py             *> out/S7_run.log 2>&1  # Step 7 图表
```

> Step 2 首次运行约 7 分钟（失败率 21.6%）；Step 2b 补救约 29 分钟（失败率 0%）。
> 若只想复现主结果，**Step 2b 可用已有 `out/feat3d_*.csv` 跳过**。

---

## 3. 产物清单

| 文件 | 内容 |
|---|---|
| `out/S0_inventory.txt` | Step 0 核对原始输出 |
| `out/S1_model_matrix.csv` | 10 个模型组合定义 |
| `out/feat3d_train.csv` / `feat3d_unknown.csv` | 3D 描述符（556 / 659 行，0 缺失） |
| `out/S2_embed_methods.csv` | 每分子的构象方法（MMFF/UFF/ETKDG_noopt） |
| `out/S3_label_sensitivity.csv` | L1–L4 敏感性结果 |
| `out/S3_ambiguous_placement.csv` | 331 个 Ambiguous 药的模型定位 |
| `out/S4_descriptor_comparison.csv` | F1–F5 的 AUC/CI |
| `out/S5_scores_all_models.xlsx` | 7 模型 × 659 药完整打分 |
| `out/S5_pairwise_spearman.csv` | 模型两两排序一致性 |
| `out/S5_jaccard_top5.csv` / `top10.csv` | Top-k% 集合 Jaccard |
| `out/S5_consensus_layers.csv` | 共识分层汇总 |
| `out/S5_discordant_ranked.csv` | 分歧药排序（主攻候选来源） |
| `out/S6_wetlab_candidates.xlsx` | 16 个湿实验候选 |
| `out/S8_findings.md` | 结论汇总 |
| `figures/FigS1..S5_*.png` | 5 张图 |

---

## 4. 关键数字（用于校验复现是否一致）

| 指标 | 期望值 |
|---|---|
| 训练集 | 556 = 205 Most + 351 No |
| 固定 split | train 444 / test 112 |
| 特征维度 | 1031 = 1024 FP + 7 desc |
| 基线 RF(500) 5-fold CV AUC | **0.7988 ± 0.0432** |
| L1 全 556 CV AUC | **0.8046** |
| F1(Morgan) CV AUC | **0.8104** |
| F4(3D) CV AUC | **0.6339** |
| ΔAUC(F5−F3) | **−0.0145** |
| Top-10% 分层 | HIGH 11 / LOW 443 / Discordant 205 |
| 3D 嵌入失败率（补救后） | **0%** |

---

## 5. 常见坑

1. **FDA 类别大小写**：`vMOST-DILI-concern`×2、`vNo-DILI-Concern`×1。必须用 `dili_common.load_fda()` 规范化，否则静默丢 3 个药。
2. **Python stdout 缓冲**：后台运行时输出可能延迟落盘，用 `tail -f` 观察而不是反复 grep。
3. **raw string 警告**：脚本 docstring 含 `\D` 会报 `SyntaxWarning`（无害，可忽略或改 `\D` 为 `\\D`）。
4. **ETKDG 参数名**：新版 RDKit 用 `params.maxIterations`，不是 `maxAttempts`（后者已是 `EmbedMolecule` 位置参数）。
5. **3D 与方法对齐**：3D 表索引是 556/659 行序，做固定 split 对比时**必须按描述符匹配取子集**，不能直接 `concat`（会得到 556 vs 444 的长度错误）。
