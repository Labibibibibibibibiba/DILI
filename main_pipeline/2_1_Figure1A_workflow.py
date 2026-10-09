import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# =========================
# basic settings
# =========================

fig, ax = plt.subplots(figsize=(6,10))

ax.axis("off")


# =========================
# function
# =========================

def add_box(text, y, height):

    box = FancyBboxPatch(
        (0.2, y),
        0.6,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        facecolor="white"
    )

    ax.add_patch(box)

    ax.text(
        0.5,
        y+height/2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        linespacing = 1.3
    )


def add_arrow(y1,y2):

    ax.annotate(
        "",
        xy=(0.5,y2),
        xytext=(0.5,y1),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2
        )
    )


# =========================
# boxes
# =========================

boxes = [

("FDA DILIrank 2.0\n1336 compounds",
0.88,
0.07),

("Excluded:\nvLess-DILI concern (n=351)\nAmbiguous-DILI concern (n=354)",
0.74,
0.10),

("Final dataset\n631 compounds\n217 DILI+ / 414 DILI-",
0.60,
0.09),

("Molecular feature extraction\n\nDescriptors:\nMW, LogP, TPSA, HBD, HBA,\nRotatable bonds, Aromatic rings\n\n1024-bit Morgan fingerprints",
0.40,
0.16),

("Machine learning models\n\nLogistic Regression\nRandom Forest\nXGBoost",
0.24,
0.13),

("Optimized Random Forest\n\nPrediction\n+\nSHAP interpretation\n+\nChemical motif discovery",
0.05,
0.15)

]


for text,y,h in boxes:
    add_box(text,y,h)


# arrows

ys=[0.88,0.76,0.64,0.48,0.32]

for i in range(len(ys)-1):
    add_arrow(
        ys[i],
        ys[i+1]+0.07
    )


# title

plt.title(
    "Figure 1A. Machine learning framework for DILI risk prediction",
    fontsize=13,
    pad=20
)


plt.tight_layout()


plt.savefig(
    "../result/Figure1A_workflow.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()