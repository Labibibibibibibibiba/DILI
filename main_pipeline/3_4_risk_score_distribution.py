import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from scipy.stats import mannwhitneyu



# ==========================
# Path
# ==========================

input_file = (
    r"D:\DILI\data\DILI_unknown_prediction_rank.xlsx"
)


output_excel = (
    r"D:\DILI\data\DILI_unknown_risk_analysis.xlsx"
)


fig_file = (
    r"D:\DILI\result\Figure4A_risk_score_distribution.png"
)



# ==========================
# Load data
# ==========================

df = pd.read_excel(
    input_file
)


print("====================")
print("Dataset")
print(df.shape)



# ==========================
# Basic statistics
# ==========================

print("\nOverall risk score statistics")

print(
    df["DILI_risk_score"]
    .describe()
)



# ==========================
# Group statistics
# ==========================

group_stats = (
    df
    .groupby("DILI_category")
    ["DILI_risk_score"]
    .describe()
)


print("\nGroup statistics")

print(group_stats)



# ==========================
# Mann Whitney U test
# ==========================


less = df[
    df["DILI_category"]
    ==
    "vLess-DILI-concern"
]["DILI_risk_score"]


ambiguous = df[
    df["DILI_category"]
    ==
    "Ambiguous-DILI-concern"
]["DILI_risk_score"]



stat, p = mannwhitneyu(
    less,
    ambiguous,
    alternative="two-sided"
)


print("\nMann-Whitney U test")

print(
    "U statistic:",
    stat
)

print(
    "p value:",
    p
)



# ==========================
# Plot distribution
# ==========================


plt.figure(
    figsize=(7,5)
)


plt.hist(
    df["DILI_risk_score"],
    bins=30
)


plt.xlabel(
    "Predicted DILI risk score"
)

plt.ylabel(
    "Number of compounds"
)


plt.title(
    "RF predicted DILI risk score distribution"
)


plt.tight_layout()


plt.savefig(
    fig_file,
    dpi=300
)


plt.show()



# ==========================
# Top percentile candidates
# ==========================


percentiles = [
    0.05,
    0.10,
    0.20
]


top_results = {}



for p in percentiles:

    n = int(
        len(df)*p
    )

    top = (
        df
        .sort_values(
            "DILI_risk_score",
            ascending=False
        )
        .head(n)
    )

    top_results[
        f"Top_{int(p*100)}%"
    ] = top



    print(
        f"\nTop {int(p*100)}%"
    )

    print(
        top[
            [
                "CompoundName",
                "DILI_category",
                "DILI_risk_score"
            ]
        ]
        .head(20)
    )



# ==========================
# Save excel
# ==========================


with pd.ExcelWriter(
    output_excel
) as writer:

    df.to_excel(
        writer,
        sheet_name="All_scores",
        index=False
    )


    group_stats.to_excel(
        writer,
        sheet_name="Group_statistics"
    )


    for name,data in top_results.items():

        data.to_excel(
            writer,
            sheet_name=name,
            index=False
        )



print("\nSaved:")
print(output_excel)