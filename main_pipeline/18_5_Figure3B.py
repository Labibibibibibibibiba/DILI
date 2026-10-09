import matplotlib.pyplot as plt
import numpy as np


# ============================
# data
# ============================

fingerprints = [
    "FP_356",
    "FP_367",
    "FP_849",
    "FP_726"
]

compounds = [
    "Phenoxybenzamine\nhydrochloride",
    "Acitretin",
    "Acetaminophen",
    "Abiraterone\nacetate"
]

OR = [
    4.86,
    4.11,
    3.16,
    2.05
]

FDR = [
    "FDR=7.41×10⁻⁵",
    "FDR=0.00146",
    "FDR=0.0261",
    "FDR=0.0293"
]


# ============================
# plot
# ============================

fig, ax = plt.subplots(figsize=(8,5))


y=np.arange(len(fingerprints))


bars=ax.barh(
    y,
    OR
)


ax.set_yticks(y)
ax.set_yticklabels(fingerprints)

ax.invert_yaxis()


ax.set_xlabel(
    "Odds ratio of DILI enrichment"
)


ax.set_title(
    "Representative compounds associated with enriched fingerprints"
)


# OR labels

for i,v in enumerate(OR):

    ax.text(
        v+0.05,
        i,
        f"{v:.2f}",
        va="center",
        fontsize=10
    )


# compound labels

for i,c in enumerate(compounds):

    ax.text(
        0.05,
        i+0.28,
        c,
        fontsize=9
    )


# FDR

for i,f in enumerate(FDR):

    ax.text(
        0.05,
        i-0.28,
        f,
        fontsize=8
    )


plt.tight_layout()


plt.savefig(
    "Figure3B_representative_compounds.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()