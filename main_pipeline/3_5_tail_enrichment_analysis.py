import pandas as pd
import numpy as np

from scipy.stats import fisher_exact



# ==========================
# Path
# ==========================

input_file = (
    r"D:\DILI\data\DILI_unknown_prediction_rank.xlsx"
)


output_file = (
    r"D:\DILI\data\DILI_tail_enrichment_analysis.xlsx"
)



# ==========================
# Load
# ==========================

df = pd.read_excel(
    input_file
)


print("====================")
print("Dataset:")
print(df.shape)



# ==========================
# Overall composition
# ==========================


total_counts = (
    df["DILI_category"]
    .value_counts()
)


print("\nOverall category:")

print(total_counts)



# ==========================
# Top tail analysis
# ==========================


percentages = [
    0.05,
    0.10,
    0.20
]


results = []



for p in percentages:


    n = int(
        len(df)*p
    )


    top_df = (
        df
        .sort_values(
            "DILI_risk_score",
            ascending=False
        )
        .head(n)
    )


    top_counts = (
        top_df["DILI_category"]
        .value_counts()
    )


    vless_top = (
        top_counts
        .get(
            "vLess-DILI-concern",
            0
        )
    )


    amb_top = (
        top_counts
        .get(
            "Ambiguous-DILI-concern",
            0
        )
    )


    # remaining

    vless_other = (
        total_counts["vLess-DILI-concern"]
        -
        vless_top
    )


    amb_other = (
        total_counts["Ambiguous-DILI-concern"]
        -
        amb_top
    )


    table = [

        [
            vless_top,
            vless_other
        ],

        [
            amb_top,
            amb_other
        ]

    ]


    oddsratio, pvalue = fisher_exact(
        table
    )


    enrichment_ratio = (
        (vless_top/n)
        /
        (total_counts["vLess-DILI-concern"]
         /
         len(df))
    )



    results.append({

        "Top_fraction":
            f"Top {int(p*100)}%",

        "N":
            n,

        "vLess_count":
            vless_top,

        "Ambiguous_count":
            amb_top,

        "vLess_fraction":
            vless_top/n,

        "Ambiguous_fraction":
            amb_top/n,

        "vLess_enrichment_ratio":
            enrichment_ratio,

        "Fisher_p":
            pvalue,

        "Odds_ratio":
            oddsratio

    })



# ==========================
# Output
# ==========================


result_df = pd.DataFrame(
    results
)


print("\n====================")
print(result_df)



result_df.to_excel(
    output_file,
    index=False
)


print("\nSaved:")
print(output_file)