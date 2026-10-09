#!/usr/bin/env python3
"""23b_fetch_shard.py <shard_idx> <n_shards>
并行抓取 PubChem 名称->SMILES 的分片工作器（每个分片写独立 csv，最后合并）。"""
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

EXT = "D:/DILI/data/external"
CACHE = f"{EXT}/pubchem_cache.csv"
SHARD = sys.argv[1]
NSHARD = int(sys.argv[2])
OUT = f"{EXT}/shard_{SHARD}.csv"

SALT = (r"\s+(sodium|potassium|hydrochloride|hcl|hydrobromide|sulfate|sulphate|mesylate|maleate|"
        r"tartrate|citrate|phosphate|acetate|dihydrate|monohydrate|hydrate|calcium|magnesium|"
        r"besylate|tosylate|fumarate|succinate|bitartrate|gluconate|bromide|chloride|iodide|"
        r"nitrate|oxide|salt|salts|dinitrate|hyclate|pamoate|embonate)$")


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(SALT, "", s)
    return re.sub(r"\s+", " ", s).strip()


def fetch(name, tries=2):
    q = urllib.parse.quote(str(name))
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/property/CanonicalSMILES/TXT"
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=12) as r:
                t = r.read().decode("utf-8", "ignore").strip().splitlines()
                return t[0].strip() if t else None
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                return None
            time.sleep(1.5)
        except Exception:
            time.sleep(1.0)
    return None


dil = pd.read_excel(f"{EXT}/DILIst.xlsx")
dil.columns = [str(c).strip() for c in dil.columns]
our = pd.read_excel("D:/DILI/data/DILI_ML_dataset_v5_final.xlsx")
our_keys = set(our["CompoundName"].map(norm))
dil["k"] = dil["CompoundName"].map(norm)
ext = dil[~dil["k"].isin(our_keys)].drop_duplicates(subset=["k"]).reset_index(drop=True)
keys = list(ext["k"])

# 已完成（主缓存 + 各分片缓存）
done = set()
if os.path.exists(CACHE):
    done |= set(pd.read_csv(CACHE)["k"].astype(str))
for i in range(NSHARD):
    p = f"{EXT}/shard_{i}.csv"
    if os.path.exists(p):
        try:
            done |= set(pd.read_csv(p)["k"].astype(str))
        except Exception:
            pass

todo = [k for k in keys if k not in done]
mine = todo[int(SHARD)::NSHARD]
print(f"[shard {SHARD}] 总 {len(keys)}，已完成 {len(done)}，本片 {len(mine)}", flush=True)

rows = []
for i, k in enumerate(mine):
    s = fetch(k)
    rows.append({"k": k, "smiles": s})
    if (i + 1) % 25 == 0:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"[shard {SHARD}] {i+1}/{len(mine)}", flush=True)
    time.sleep(0.8)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"[shard {SHARD}] done {len(rows)}", flush=True)
