#!/usr/bin/env python3
"""
24_rescore_unknown_v2.py
========================
用 v2 模型重算 659 个未分类药物的风险分数、尾部富集与验证药位置。

输出：D:/DILI/data/DILI_unknown_prediction_rank_v2.xlsx
      D:/DILI/data/DILI_tail_enrichment_analysis_v2.xlsx
      D:/DILI/data/unknown_summary_v2.json
"""

import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

DATA = "D:/DILI/data"
LOG = []


def log(m=""):
    print(m, flush=True)
    LOG.append(str(m))


def or_ci(a, b, c, d):
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    o = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(o), float(np.exp(np.log(o) - 1.96 * se)), float(np.exp(np.log(o) + 1.96 * se))


df = pd.read_excel(f"{DATA}/DILI_unknown_ML_matrix.xlsx")
model = joblib.load(f"{DATA}/best_RF_model_v2.pkl")

desc = ["MolecularWeight", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "AromaticRings"]
fps = [f"FP_{i}" for i in range(1024)]
feat = desc + fps
missing = [c for c in feat if c not in df.columns]
if missing:
    raise ValueError(f"缺特征: {missing[:5]} ...")
log(f"[1] 未分类药 {df.shape}；特征 {len(feat)} 全部匹配")

score = model.predict_proba(df[feat])[:, 1]
res = df[["LTKBID", "CompoundName", "DILI_category", "SMILES"]].copy()
res["DILI_risk_score"] = score
res = res.sort_values("DILI_risk_score", ascending=False).reset_index(drop=True)
res.insert(0, "Rank", range(1, len(res) + 1))
res.to_excel(f"{DATA}/DILI_unknown_prediction_rank_v2.xlsx", index=False)

n = len(res)
log(f"    分数范围 [{score.min():.4f}, {score.max():.4f}]  mean={score.mean():.4f}  std={score.std():.4f}")
log(f"    类别构成: {res['DILI_category'].value_counts().to_dict()}")

# ---- vLess vs Ambiguous ----
vl = res.loc[res["DILI_category"] == "vLess-DILI-concern", "DILI_risk_score"]
am = res.loc[res["DILI_category"] == "Ambiguous-DILI-concern", "DILI_risk_score"]
u, p_mw = mannwhitneyu(vl, am, alternative="two-sided")
log(f"[2] vLess n={len(vl)} mean={vl.mean():.4f} | Ambiguous n={len(am)} mean={am.mean():.4f} "
    f"| Mann-Whitney U p = {p_mw:.4f}")

# ---- 尾部富集 ----
tot = res["DILI_category"].value_counts()
rows = []
for p in [0.05, 0.10, 0.20]:
    k = int(n * p)
    top = res.head(k)
    tc = top["DILI_category"].value_counts()
    vt = int(tc.get("vLess-DILI-concern", 0))
    at = int(tc.get("Ambiguous-DILI-concern", 0))
    vo = int(tot["vLess-DILI-concern"]) - vt
    ao = int(tot["Ambiguous-DILI-concern"]) - at
    o, pv = fisher_exact([[vt, vo], [at, ao]])
    o2, lo, hi = or_ci(vt, vo, at, ao)
    rows.append({"Top_fraction": f"Top {int(p*100)}%", "N": k, "vLess_count": vt,
                 "Ambiguous_count": at, "vLess_fraction": vt / k,
                 "Ambiguous_fraction": at / k,
                 "vLess_enrichment_ratio": (vt / k) / (tot["vLess-DILI-concern"] / n),
                 "Fisher_p": pv, "Odds_ratio": o,
                 "OR_CI_low": lo, "OR_CI_high": hi,
                 "boundary_score": float(res.iloc[k - 1]["DILI_risk_score"]),
                 "boundary_drug": res.iloc[k - 1]["CompoundName"]})
tail = pd.DataFrame(rows)
tail.to_excel(f"{DATA}/DILI_tail_enrichment_analysis_v2.xlsx", index=False)
log("\n[3] 尾部富集:")
log(tail[["Top_fraction", "N", "vLess_count", "Ambiguous_count", "vLess_fraction",
          "Fisher_p", "Odds_ratio", "OR_CI_low", "OR_CI_high",
          "boundary_score", "boundary_drug"]].to_string(index=False))

# ---- 验证药位置 ----
drugs = ["Chlorpromazine", "Sulfamethizole", "Cabotegravir", "Mebendazole",
         "Oxybutynin", "Tolterodine", "Metformin", "Diclofenac"]
log("\n[4] 验证药在 659 排序中的位置:")
hit_rows = []
for d in drugs:
    m = res[res["CompoundName"].str.contains(d, case=False, na=False)]
    if len(m):
        for _, r in m.iterrows():
            hit_rows.append({"drug": d, "matched": r["CompoundName"], "rank": int(r["Rank"]),
                             "score": float(r["DILI_risk_score"]), "category": r["DILI_category"]})
            log(f"    {d:16s} → rank {int(r['Rank']):3d}/659  score={r['DILI_risk_score']:.4f}  {r['DILI_category']}")
    else:
        log(f"    {d:16s} → 未在 659 未分类集中找到")
pd.DataFrame(hit_rows).to_excel(f"{DATA}/validation_drug_positions_v2.xlsx", index=False)

summary = {"n": n, "score_mean": float(score.mean()), "score_std": float(score.std()),
           "score_max": float(score.max()),
           "mannwhitney_p": float(p_mw),
           "vless_mean": float(vl.mean()), "ambig_mean": float(am.mean()),
           "tail": tail.to_dict("records"),
           "validation_drugs": hit_rows}
json.dump(summary, open(f"{DATA}/unknown_summary_v2.json", "w"), indent=2, ensure_ascii=False, default=str)
open(f"{DATA}/24_rescore_log.txt", "w", encoding="utf-8").write("\n".join(LOG))
log("\n[5] 完成")
