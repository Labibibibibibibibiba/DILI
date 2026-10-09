#!/usr/bin/env python3
"""23c_fetch_fast.py — 线程池 + 连接复用 抓取 PubChem 名称->SMILES。
全局速率限制 ~4 req/s（PubChem 上限 5/s）。"""
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

EXT = "D:/DILI/data/external"
CACHE = f"{EXT}/pubchem_cache.csv"
N_THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
RATE = 4.0

SALT = (r"\s+(sodium|potassium|hydrochloride|hcl|hydrobromide|sulfate|sulphate|mesylate|maleate|"
        r"tartrate|citrate|phosphate|acetate|dihydrate|monohydrate|hydrate|calcium|magnesium|"
        r"besylate|tosylate|fumarate|succinate|bitartrate|gluconate|bromide|chloride|iodide|"
        r"nitrate|oxide|salt|salts|dinitrate|hyclate|pamoate|embonate)$")


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(SALT, "", s)
    return re.sub(r"\s+", " ", s).strip()


_lock = threading.Lock()
_next_slot = [time.time()]


def rate_limit():
    with _lock:
        now = time.time()
        if _next_slot[0] < now:
            _next_slot[0] = now
        wait = _next_slot[0] - now
        _next_slot[0] += 1.0 / RATE
    if wait > 0:
        time.sleep(wait)


_tls = threading.local()


def sess():
    if not hasattr(_tls, "s"):
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (research; DILI-structure-analysis)"})
        _tls.s = s
    return _tls.s


def fetch(name):
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{requests.utils.quote(str(name))}/property/CanonicalSMILES/TXT")
    for _ in range(2):
        rate_limit()
        try:
            r = sess().get(url, timeout=12)
            if r.status_code == 200:
                t = r.text.strip().splitlines()
                return t[0].strip() if t else None
            if r.status_code in (404, 400):
                return None
            if r.status_code == 503:
                time.sleep(2.0)
                continue
        except Exception:
            time.sleep(0.8)
    return None


dil = pd.read_excel(f"{EXT}/DILIst.xlsx")
dil.columns = [str(c).strip() for c in dil.columns]
our = pd.read_excel("D:/DILI/data/DILI_ML_dataset_v5_final.xlsx")
our_keys = set(our["CompoundName"].map(norm))
dil["k"] = dil["CompoundName"].map(norm)
ext = dil[~dil["k"].isin(our_keys)].drop_duplicates(subset=["k"]).reset_index(drop=True)

done = set()
if os.path.exists(CACHE):
    done |= set(pd.read_csv(CACHE)["k"].astype(str))
todo = [k for k in ext["k"] if k not in done]
print(f"总 {len(ext)}，已完成 {len(done)}，待抓 {len(todo)}", flush=True)

rows = list(pd.read_csv(CACHE).to_dict("records")) if os.path.exists(CACHE) else []
buf = []
t0 = time.time()
with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
    for i, (k, s) in enumerate(zip(todo, ex.map(fetch, todo))):
        buf.append({"k": k, "smiles": s})
        if (i + 1) % 100 == 0:
            pd.DataFrame(rows + buf).drop_duplicates(subset=["k"]).to_csv(CACHE, index=False)
            el = time.time() - t0
            print(f"  {i+1}/{len(todo)}  用时 {el:.0f}s  ({ (i+1)/el:.1f}/s )", flush=True)
pd.DataFrame(rows + buf).drop_duplicates(subset=["k"]).to_csv(CACHE, index=False)
print("done", len(rows) + len(buf), flush=True)
