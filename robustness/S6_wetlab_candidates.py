# -*- coding: utf-8 -*-
"""
Step 6 — Part D: PHH 湿实验候选清单
产出: out/S6_wetlab_candidates.xlsx, out/S6_wetlab_rationale.md, out/S6_step6.log

规则(交接文件 §5 Step6):
  - Discordant 组(主攻): 按 |Δ score| 降序 + 化学多样性 + 可得性 -> n=8~12
  - 锚点: Consensus_HIGH 2~3 个 (阳性对照); Consensus_LOW 2~3 个 (阴性对照)
  - 与 D:\work space\DILI_PHH_阴性对照面板.xlsx 去重合并
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dili_common as dc

log = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.append(s)

P("=" * 72)
P("Step 6 — Part D PHH 湿实验候选清单")
P("=" * 72)

scores = pd.read_csv(dc.OUT / "S5_high_risk_sets.csv")
score_cols = [c for c in scores.columns if c.startswith("L1_")]
P(f"\n[6.1] 载入打分表 {scores.shape}, 模型列 {len(score_cols)}")

# ======================================================================
# 1. 已有阴性对照面板(去重参考)
# ======================================================================
known = set()
for p in [Path(r"D:\work space\DILI_PHH_阴性对照面板.xlsx"),
          Path(r"D:\work space\DILI_对照药面板_建议V2.2.xlsx")]:
    if p.exists():
        try:
            k = pd.read_excel(p)
            P(f"[6.2] {p.name}: {len(k)} 行, 列 = {list(k.columns)}")
            # 优先找真正的药物名/化合物名列
            cand_cols = [c for c in k.columns if any(
                t in str(c).lower() for t in ["compound", "name", "药", "化合物", "药物", "smiles"])]
            if cand_cols:
                col = cand_cols[0]
                known |= set(k[col].dropna().astype(str).str.strip())
                P(f"[6.2]   使用列 '{col}', 收集 {k[col].notna().sum()} 个名称")
            else:
                # 逐列试探: 含字母且非"对照/角色/层级"字样的列
                for c in k.columns:
                    vals = k[c].dropna().astype(str)
                    if vals.str.contains("对照|角色|层级|候选|基准|可选|必选|保留|平行").mean() < 0.5 \
                       and vals.str.match(r"[A-Za-z]").mean() > 0.5:
                        known |= set(vals.str.strip())
                        P(f"[6.2]   使用列 '{c}' (启发式)")
                        break
        except Exception as e:
            P(f"[6.2] [WARN] 读 {p.name} 失败: {e}")
P(f"[6.2] 已知对照药 {len(known)} 个: {sorted(known)[:12]}")

# ======================================================================
# 2. Discordant 主攻组
# ======================================================================
def chemical_pick(df, n):
    """按跨模型分数 SD 降序 + Butina 聚类(ECFP4, Tanimoto 0.7)保证多样性。"""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    from rdkit.ML.Cluster import Butina
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    df = df.copy()
    df["_sd"] = df[score_cols].std(axis=1)
    df = df.sort_values("_sd", ascending=False).reset_index(drop=True)

    fps, keep = [], []
    for i, r in df.iterrows():
        mol = Chem.MolFromSmiles(str(r.get("SMILES", "")))
        if mol is None:
            continue
        fps.append(GetMorganFingerprintAsBitVect(mol, 2, 2048))
        keep.append(i)
    if not fps:
        return df.head(n)

    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    clusters = Butina.ClusterData(dists, len(fps), 0.7, isDistData=True)

    picked = []
    for cl in sorted(clusters, key=lambda c: min(keep[j] for j in c)):
        # 每个簇取 SD 最高的一个
        best = min(cl, key=lambda j: keep[j])
        picked.append(keep[best])
        if len(picked) >= n:
            break
    return df.iloc[picked]


disc = scores[scores["layer"] == "Discordant"].copy()
P(f"\n[6.3] Discordant 池 n={len(disc)}")

main_pick = chemical_pick(disc, 10)
main_pick["role"] = "Discordant(主攻)"

# 锚点
cons_hi = scores[scores["layer"] == "Consensus_HIGH"].copy()
cons_hi["_sd"] = cons_hi[score_cols].std(axis=1)
cons_hi = cons_hi.sort_values(["_sd"], ascending=True).head(3).copy()   # 一致性最高
cons_hi["role"] = "Consensus_HIGH(阳性锚点)"
cons_lo = scores[scores["layer"] == "Consensus_LOW"].copy()
cons_lo["_sd"] = cons_lo[score_cols].std(axis=1)
cons_lo = cons_lo.sort_values(["n_high", "_sd"]).head(3).copy()
cons_lo["role"] = "Consensus_LOW(阴性锚点)"

panel = pd.concat([main_pick, cons_hi, cons_lo], ignore_index=True)
panel["is_known_ctrl"] = panel["CompoundName"].astype(str).str.strip().isin(known)
P(f"[6.4] 组合面板 {len(panel)} 个")
P(panel[["CompoundName", "DILI_category", "role", "n_high", "is_known_ctrl"]].to_string(index=False))

# ======================================================================
# 3. 附文献剂量/暴露(留空待填) + 可选性说明
# ======================================================================
panel["Cmax_ref_uM"] = np.nan
panel["dose_ref"] = ""
panel["availability"] = ""
panel["rationale"] = panel.apply(
    lambda r: (f"跨模型分数 SD={r[[c for c in score_cols]].std():.3f}; "
               f"高风险票数 {int(r['n_high'])}/{len(score_cols)}; "
               f"类别 {r['DILI_category']}"), axis=1)

out_cols = ["CompoundName", "DILI_category", "SMILES", "role",
            "n_high", "is_known_ctrl", "Cmax_ref_uM", "dose_ref",
            "availability", "rationale"] + score_cols
panel_out = panel[[c for c in out_cols if c in panel.columns]]
panel_out.to_excel(dc.OUT / "S6_wetlab_candidates.xlsx", index=False)
P(f"\n[6.5] 候选清单 -> out/S6_wetlab_candidates.xlsx  ({len(panel_out)} 行)")

# ======================================================================
# 4. 理由文档
# ======================================================================
md = ["# PHH 湿实验候选清单 — 理由说明", "",
      f"生成日期: 2026-10-04 | 来源: S5 跨模型打分 (Top-10% 共识分层)", "",
      "## 设计原则", "",
      "- **主攻（Discordant）**：模型分歧最大 → 结构信息不足以判定 → 湿实验价值最高",
      "- **阳性锚点（Consensus_HIGH）**：所有模型一致判高风险 → 验证实验体系灵敏度",
      "- **阴性锚点（Consensus_LOW）**：所有模型一致判低风险 → 验证实验体系特异性",
      "- **读数设计**：PHH 剂量-反应 IC50 与 Cmax 比值（对标 Oakuloid iBAC 设计）", "",
      "## 候选表", ""]
md.append("| # | 药物 | 类别 | 角色 | 高风险票数 | 跨模型SD | 已在对照面板 |")
md.append("|---|---|---|---|---|---|---|")
for i, (_, r) in enumerate(panel_out.iterrows(), 1):
    sd = panel.loc[panel["CompoundName"] == r["CompoundName"], score_cols].std(axis=1).values
    sd = sd[0] if len(sd) else np.nan
    md.append(f"| {i} | {r['CompoundName']} | {r['DILI_category']} | {r['role']} | "
              f"{int(r['n_high'])}/{len(score_cols)} | {sd:.3f} | {'是' if r['is_known_ctrl'] else '否'} |")
md += ["", "## 待补", "",
       "- [ ] 每个候选药的 Cmax / 临床剂量（文献回查，写入 `Cmax_ref_uM` / `dose_ref`）",
       "- [ ] 商业可得性（货号、纯度、溶解性）",
       "- [ ] 若与既有阴性对照面板重复，需说明差额"]
(Path(dc.OUT / "S6_wetlab_rationale.md")).write_text("\n".join(md), encoding="utf-8")
P(f"[6.6] 理由文档 -> out/S6_wetlab_rationale.md")

(json.dumps(dict(n_panel=len(panel_out),
                 n_discordant=int((panel["role"].str.startswith("Discordant")).sum()),
                 n_pos_anchor=int((panel["role"].str.startswith("Consensus_HIGH")).sum()),
                 n_neg_anchor=int((panel["role"].str.startswith("Consensus_LOW")).sum())),
            ensure_ascii=False, indent=2) )
(dc.OUT / "S6_step6.log").write_text("\n".join(log), encoding="utf-8")
P("\nStep 6 完成。")
