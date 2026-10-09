import importlib, sys
mods = ["numpy","pandas","openpyxl","sklearn","rdkit","joblib","scipy","matplotlib","xgboost","lightgbm","catboost"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"MISS {m:12s} {type(e).__name__}")
print("python", sys.version.split()[0], sys.executable)
