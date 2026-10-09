import pandas as pd


df=pd.read_excel(
r"D:\DILI\data\DILI_ML_dataset_v5_final.xlsx"
)


for fp in [
    "FP_356",
    "FP_367",
    "FP_849"
]:

    print("================")
    print(fp)

    print(
        df[df[fp]==1][
            ["CompoundName","DILI_label"]
        ].head(5)
    )

    print(
        "count:",
        df[fp].sum()
    )