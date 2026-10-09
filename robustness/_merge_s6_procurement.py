# -*- coding: utf-8 -*-
"""把备份 S6 的采购列(cas_no/procurement_tier/supplier_ref/price_ref)合并回修复后的 S6。
   类别(DILI_category)一律用修复值(新表), 采购列用备份值。"""
import pandas as pd

new = pd.read_excel(r"D:/DILI_1/out/S6_wetlab_candidates.xlsx")
bak = pd.read_excel(r"D:/DILI_1/out/_BUGGY_backup_S6_wetlab_candidates.xlsx")

proc_cols = ["cas_no", "procurement_tier", "supplier_ref", "price_ref"]
bak_key = bak.set_index(bak["CompoundName"].astype(str).str.strip())
for c in proc_cols:
    m = bak_key[c].to_dict()
    new[c] = new["CompoundName"].astype(str).str.strip().map(m)

# 重排序: 把采购列放到 role 前面
order = ["CompoundName", "DILI_category"] + proc_cols + [
    "role", "n_high", "is_known_ctrl", "Cmax_ref_uM", "dose_ref",
    "availability", "rationale",
    "L1_F1_A1", "L1_F3_A1", "L1_F2_A1", "L1_F3_A2", "L1_F3_A3", "L1_F4_A1", "L1_F5_A1"]
new = new[[c for c in order if c in new.columns]]
new.to_excel(r"D:/DILI_1/out/S6_wetlab_candidates.xlsx", index=False)

print("合并完成, 列:", list(new.columns))
print("采购列缺失值:", new[proc_cols].isna().sum().to_dict())
print(new[["CompoundName", "DILI_category", "procurement_tier"]].to_string(index=False))
