import pandas as pd


# ==========================
# Path
# ==========================

input_file = r"D:\DILI\data\Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx"

output_file = r"D:\DILI\data\DILI_unknown_compounds_raw.xlsx"



# ==========================
# Read FDA DILIrank
# ==========================

df = pd.read_excel(
    input_file,
    header=1
)


print("====================")
print("Dataset shape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())



# ==========================
# Filter
# ==========================

unknown_categories = [
    "vLess-DILI-concern",
    "Ambiguous-DILI-concern"
]


unknown_df = df[
    df["vDILI-Concern"].isin(
        unknown_categories
    )
].copy()



print("====================")

print(
    "Unknown cohort:",
    unknown_df.shape
)


print(
    unknown_df["vDILI-Concern"]
    .value_counts()
)



# ==========================
# Select columns
# ==========================

unknown_df = unknown_df[
    [
        "LTKBID",
        "CompoundName",
        "SeverityClass",
        "LabelSection",
        "vDILI-Concern"
    ]
]



unknown_df = unknown_df.rename(
    columns={
        "vDILI-Concern":
        "DILI_category"
    }
)



# ==========================
# Save
# ==========================

unknown_df.to_excel(
    output_file,
    index=False
)



print("====================")
print("Saved:")
print(output_file)


print(
    unknown_df.head()
)