# PHH 湿实验候选清单 — 理由说明

生成日期: 2026-10-05 (2026-10-05 修复行序错位 bug 后重生成) | 来源: S5 跨模型打分 (Top-10% 共识分层)

> ⚠️ **本表类别列为修复值**。2026-10-05 起重生成，修正了 `DILI_unknown_prediction_rank.xlsx`（按风险分降序）
> 与 `DILI_unknown_ML_matrix.xlsx`（按字母序）行序不同导致的**位置对齐 bug**。面板成员未变。
> **重要澄清**：这**不是改动药物标签**，而是**修复了旧代码把相邻行药物的标签错误贴到该药身上**的 bug。
> 修复后的类别与 **DILIrank 2.0 官方原表**逐条一致（经 `Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx` 核实）：
> Amlodipine/Zonisamide/Pyrazinamide/Tinidazole/Mebendazole = **vLess**（官方值）；
> Vigabatrin/Plecanatide/Peramivir/Degarelix/Eletriptan = **Ambiguous**（官方值）。
> 其中 **Mebendazole 的官方标签自始为 vLess**，旧代码曾错贴为 Ambiguous。详见 `S12_roworder_bug_fix.md`。

## 设计原则

- **主攻（Discordant）**：模型分歧最大 → 结构信息不足以判定 → 湿实验价值最高
- **阳性锚点（Consensus_HIGH）**：所有模型一致判高风险 → 验证实验体系灵敏度
- **阴性锚点（Consensus_LOW）**：所有模型一致判低风险 → 验证实验体系特异性
- **读数设计**：PHH 剂量-反应 IC50 与 Cmax 比值（对标 Oakuloid iBAC 设计）

## 候选表

| # | 药物 | 类别 | 角色 | 高风险票数 | 跨模型SD | 已在对照面板 |
|---|---|---|---|---|---|---|
| 1 | Amlodipine besylate | vLess | Discordant(主攻) | 1/7 | 0.319 | 否 |
| 2 | Vortioxetine hydrobromide | vLess | Discordant(主攻) | 4/7 | 0.308 | 否 |
| 3 | Zonisamide | vLess | Discordant(主攻) | 3/7 | 0.297 | 否 |
| 4 | Vigabatrin | Ambiguous | Discordant(主攻) | 1/7 | 0.295 | 否 |
| 5 | Plecanatide | Ambiguous | Discordant(主攻) | 1/7 | 0.294 | 否 |
| 6 | Peramivir | Ambiguous | Discordant(主攻) | 1/7 | 0.289 | 否 |
| 7 | Pyrazinamide | vLess | Discordant(主攻) | 2/7 | 0.287 | 否 |
| 8 | Tinidazole | vLess | Discordant(主攻) | 2/7 | 0.285 | 否 |
| 9 | Degarelix acetate | Ambiguous | Discordant(主攻) | 1/7 | 0.282 | 否 |
| 10 | Pyrimethamine | vLess | Discordant(主攻) | 5/7 | 0.278 | 否 |
| 11 | Sulfadiazine | vLess | Consensus_HIGH(阳性锚点) | 6/7 | 0.063 | 否 |
| 12 | Mebendazole | vLess | Consensus_HIGH(阳性锚点) | 6/7 | 0.072 | 否 |
| 13 | Dapsone | vLess | Consensus_HIGH(阳性锚点) | 6/7 | 0.084 | 否 |
| 14 | Eletriptan hydrobromide | Ambiguous | Consensus_LOW(阴性锚点) | 0/7 | 0.040 | 否 |
| 15 | Selenious acid | Ambiguous | Consensus_LOW(阴性锚点) | 0/7 | 0.048 | 否 |
| 16 | Mefloquine hydrochloride | vLess | Consensus_LOW(阴性锚点) | 0/7 | 0.051 | 否 |

## 待补

- [ ] 每个候选药的 Cmax / 临床剂量（文献回查，写入 `Cmax_ref_uM` / `dose_ref`）
- [ ] 商业可得性（货号、纯度、溶解性）
- [ ] 若与既有阴性对照面板重复，需说明差额