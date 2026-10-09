import pandas as pd
import joblib
import numpy as np


# ==========================
# Path
# ==========================

input_file = (
    r"D:\DILI\data\DILI_unknown_ML_matrix.xlsx"
)


model_file = (
    r"D:\DILI\data\best_RF_model.pkl"
)


output_file = (
    r"D:\DILI\data\DILI_unknown_prediction_rank.xlsx"
)



# ==========================
# Load data
# ==========================

df = pd.read_excel(
    input_file
)


print("====================")
print("Unknown dataset:")
print(df.shape)



# ==========================
# Load model
# ==========================

model = joblib.load(
    model_file
)


print("====================")
print("Model loaded")



# ==========================
# Identify features
# ==========================

descriptor_features = [

    "MolecularWeight",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "AromaticRings"

]


fp_features = [

    f"FP_{i}"
    for i in range(1024)

]


feature_columns = (
    descriptor_features
    +
    fp_features
)



print("====================")
print(
    "Expected features:",
    len(feature_columns)
)



# ==========================
# Check features
# ==========================

missing_features = [

    col for col in feature_columns
    if col not in df.columns

]


if len(missing_features) > 0:

    print(
        "Missing features:"
    )

    print(
        missing_features
    )

    raise ValueError(
        "Feature mismatch!"
    )


print(
    "Feature check passed"
)



# ==========================
# Prepare X
# ==========================


X_unknown = df[
    feature_columns
]



print(
    "X shape:",
    X_unknown.shape
)



# ==========================
# Prediction
# ==========================


risk_score = model.predict_proba(
    X_unknown
)[:,1]



# ==========================
# Combine result
# ==========================


result = df[
    [
        "LTKBID",
        "CompoundName",
        "DILI_category",
        "SMILES"
    ]
].copy()



result["DILI_risk_score"] = (
    risk_score
)



# ranking

result = result.sort_values(
    by="DILI_risk_score",
    ascending=False
)


result.insert(
    0,
    "Rank",
    range(
        1,
        len(result)+1
    )
)



# ==========================
# Save
# ==========================


result.to_excel(
    output_file,
    index=False
)



print("====================")
print("Prediction finished")


print(
    result.head(20)
)


print(
    "Saved:"
)

print(
    output_file
)
