# S12 — 行序错位 bug 修复报告

发现日期：2026-10-05 | 修复脚本：`_scripts/S3_label_sensitivity.py`、`_scripts/S5_cross_model_sets.py`
受影响产物：S3、S5、S6（S11 不受影响，见 §4）

---

## 一、bug 描述

`D:\DILI\data\` 下三个未知集（659 药）表格的**行序互不相同**：

| 文件 | 排序方式 | 第一行 |
|---|---|---|
| `DILI_unknown_prediction_rank.xlsx` | **按 DILI_risk_score 降序** | Ofloxacin |
| `DILI_unknown_ML_matrix.xlsx` | **按字母序** | Abemaciclib |
| `DILI_unknown_compounds_raw.xlsx` | **按字母序**（含 Abatacept，比 matrix 多一行无 SMILES） | Abatacept |

S3 第 62 行、S5 第 59–63 行原用 `.reset_index(drop=True)` 按**位置**把 `DILI_category` 贴到候选池上。

**实测后果（关键）**：

```
位置对齐下名字相同的行:        2 / 659
位置对齐下"类别值"恰好相同的行:   0 / 659
```

即**按位置取到的类别与药物完全无关联**——等价于随机贴标签。此前观测到的"320/659 错、7/16 对"是**类别值层面的巧合**（例如错位后某行仍落在 `vLess` 上），真正的关联度是 0。

抽样（mat 行名 ｜ pr 同位置行名 ｜ pr 给该位置的类别）：

```
#  0 Abemaciclib          | Ofloxacin                    | vLess
#  2 Acamprosate calcium  | Sulfamethizole               | vLess
#658 Zonisamide          | Lutetium lu 177 dotatate     | Ambiguous
```

---

## 二、修复方法

**一律改为按 `CompoundName` 连接**，并加断言防止静默失败：

```python
pr = pd.read_excel(dc.DILI / "data" / "DILI_unknown_prediction_rank.xlsx")
pr["_key"] = pr["CompoundName"].astype(str).str.strip()
mat["_key"] = mat["CompoundName"].astype(str).str.strip()
_meta = pr[["_key", "DILI_category", "SMILES"]].drop_duplicates("_key")
cand = mat.merge(_meta, on="_key", how="left", suffixes=("", "_pr"))
assert cand["DILI_category"].notna().all()   # 任何未匹配立即报错
```

> **规范**：凡跨这三张表取数，**必须按 `CompoundName` 连接，禁止按位置 `reindex`/`reset_index`**。

---

## 三、修复前后对照

### 3.1 候选池类别计数

| | 修复前（错） | 修复后（权威） |
|---|---|---|
| vLess | 327 | **328** |
| Ambiguous | 332 | **331** |

### 3.2 S3 标签敏感性 AUC

| 方案 | 修复前（作废） | 修复后（生效） |
|---|---|---|
| L1 RF | 0.8046 | 0.8046（不变） |
| L2 RF | 0.7709 | **0.7797** |
| L3 RF(P_Most) | 0.7802 | **0.7699** |
| L4 最优 | 0.8153（λ=0.6, μ=0.5） | **0.8171（λ=0.7, μ=0.6）** |

> 说明：CV 在 556/884 训练池上做，类标错位主要污染 **未知池** 的类别，故 L1 不变；
> L2–L4 因把 unknown 类别并入训练而受影响。**L4 最优折叠参数已改变**。

### 3.3 S3 Ambiguous 定位（副产品）

| | 修复前 | 修复后 |
|---|---|---|
| Ambiguous argmax 分布 | Less 95 / No 89 / Amb 85 / Most 63 | **Amb 114 / No 87 / Less 76 / Most 54** |

> 修复后 **114/331（34%）Ambiguous 药被模型判为 Ambiguous**（旧版随机贴标下仅 85）——
> 这是「Ambiguous 应单列」的**更强证据**（模型自身在无监督语义上也倾向于把它们单列）。

### 3.4 S5 共识分层

| 分层（Top-10%） | 修复前 | 修复后 |
|---|---|---|
| Consensus_HIGH | 11 | **11**（10 vLess + 1 Ambiguous） |
| Consensus_LOW | 443 | **443** |
| Discordant | 205 | **205** |

> 分层数字不变（只依赖 7 模型分数排名，与类别无关）；变的是每层的 vLess/Amb 构成。

### 3.5 S6 候选面板类别更正（**面板成员不变，仍 16 个**）

> **⚠️ 措辞澄清（2026-10-05 复核后补）**：下表右列**不是「本研究改动了药物标签」**，而是**「修复了旧代码把相邻行的药物标签错误贴到该药身上」**。所有类别均以 **DILIrank 2.0 官方原表**（`Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx`，1336 行）为准，本工作**未修改任何原始标签**。详见 §3.6 的官方原表逐条核实。

| 药物 | 旧代码错贴的类别 | 修复后（= DILIrank 2.0 官方原表） |
|---|---|---|
| Amlodipine besylate | Ambiguous | **vLess** |
| Zonisamide | Ambiguous | **vLess** |
| Pyrazinamide | Ambiguous | **vLess** |
| Tinidazole | Ambiguous | **vLess** |
| **Mebendazole** | Ambiguous | **vLess** ✅ 与 V2.2 一致 |
| Vigabatrin | vLess | **Ambiguous** |
| Plecanatide | vLess | **Ambiguous** |
| Peramivir | vLess | **Ambiguous** |
| Degarelix acetate | vLess | **Ambiguous** |
| Eletriptan hydrobromide | vLess | **Ambiguous** |
| Sulfadiazine / Dapsone | vLess | vLess（本就正确） |
| Selenious acid | Ambiguous | Ambiguous（本就正确） |
| Mefloquine hydrochloride | vLess | vLess（本就正确） |

**更正后的内部一致性校验 ✅**：3 个阳性锚点（Sulfadiazine/Mebendazole/Dapsone，各 6/7 票）**全部为 vLess**——
符合"阳性锚点应为真阳性"的逻辑；旧代码错贴时有一个 Ambiguous 混入，正是 bug 的直接症状。

### 3.6 回到 DILIrank 2.0 官方原表逐条核实（2026-10-05 补充）

**核实方法**：直接读取 `D:\DILI\data\Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx` 的 `version 2` 工作表（1336 行），按 `CompoundName` 精确匹配，取 `vDILI-Concern` 列。

**核实结果（面板相关药）**：

| 药物 | DILIrank 2.0 官方 `vDILI-Concern` | 与修复后一致？ |
|---|---|---|
| Mebendazole | **vLess-DILI-concern** | ✅ |
| Cabotegravir sodium | **Ambiguous-DILI-concern** | ✅ |
| Sulfamethizole | vLess-DILI-concern | ✅ |
| Chlorpromazine hydrochloride | vLess-DILI-concern | ✅ |
| Oxybutynin chloride | vLess-DILI-concern | ✅ |
| Tolterodine tartrate | vLess-DILI-concern | ✅ |
| Metformin hydrochloride | vLess-DILI-concern | ✅ |
| Diclofenac（sodium） | vMost-DILI-concern | ✅（池外基准，符合预期） |

**Mebendazole 个案链条（完整可复现）**：

1. **DILIrank 2.0 官方原表**：`LT00308 | Mebendazole | Severity 3 | Warnings & precautions | vLess-DILI-concern | Unchanged` → **vLess**
2. **中间表 `DILI_unknown_ML_matrix.xlsx` / `_compounds_raw.xlsx`**：Mebendazole 行自带的 `DILI_category` 列也写 **vLess**（与官方一致）
3. **`DILI_unknown_prediction_rank.xlsx`**：Mebendazole 位于**第 12 行**（按风险分降序），其类别列也是 vLess
4. **旧代码错在哪**：它用 `pr` 表的**第 358 行**（Mebendazole 在 `mat` 表中的行号）去贴类别 —— 而 `pr` 表第 358 行是 **`Pralidoxime chloride`（Ambiguous）**。于是 Mebendazole 被错误地贴上 Ambiguous。
5. **修复**：改为按 `CompoundName` join → 取回 Mebendazole 自己的 vLess。

> **一句话**：Mebendazole **从来就是 vLess**；出现的 "Ambiguous" 完全是旧代码跨表按位置贴标签产生的假象。本工作**没有、也不需要修改任何药物标签**。

---

## 四、未受影响 / 已备份

- **S11 不受影响**：`S11_label_rank_stability.py` 自始即按 `CompoundName` join，结论（V2.2 八药在 36 组合下的排名）**干净可用**。
- **S1/S2/S4/S7–S10 不受影响**：不触碰未知池类别列。
- **备份**：全部受污染产物已存为 `out/_BUGGY_backup_*`（11 个文件），**任何旧数值不得再引用**。

---

## 五、代码坑记录（新增）

1. **三表行序不同**（本 bug）：见 §2 规范。
2. `DILI_unknown_compounds_raw.xlsx` 比 `_ML_matrix.xlsx` **多 1 行**（Abatacept，无 SMILES），行数与行序都不同——不能假设两者可互换。
3. 修复 `merge` 后 `SMILES` 会产生 `SMILES` / `SMILES_pr` 双列，下游取数须 coalesce，否则可能取到全 NaN 的左列。
