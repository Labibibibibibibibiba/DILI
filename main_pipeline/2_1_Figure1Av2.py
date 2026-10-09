import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# =========================
# Figure settings
# =========================

fig, ax = plt.subplots(
    figsize=(5, 9.5,)
)

ax.axis("off")


# =========================
# Box function
# =========================

def add_box(text, y, height):

    box = FancyBboxPatch(
        (0.18, y),
        0.64,
        height,
        boxstyle="round,pad=0.025",
        linewidth=1.2,
        facecolor="white"
    )

    ax.add_patch(box)

    ax.text(
        0.5,
        y + height/2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        linespacing=1.25
    )


# =========================
# Arrow function
# =========================

def add_arrow(y_start, y_end):

    ax.annotate(
        "",
        xy=(0.5, y_end),
        xytext=(0.5, y_start),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
            shrinkA=0,
            shrinkB=0
        )
    )


# =========================
# Workflow content
# =========================

boxes = [

    (
        "FDA DILIrank 2.0\n"
        "n = 1336 compounds",
        0.86,
        0.08
    ),

    (
        "Data curation\n"
        "Remove vLess- and\n"
        "Ambiguous-DILI compounds",
        0.70,
        0.10
    ),

    (
        "Final dataset\n"
        "631 drugs\n"
        "217 DILI+  |  414 DILI-",
        0.54,
        0.10
    ),

    (
        "Molecular representation\n"
        "Physicochemical descriptors\n"
        "Morgan fingerprints",
        0.36,
        0.12
    ),

    (
        "Machine learning models\n"
        "LR  |  RF  |  XGBoost",
        0.22,
        0.08
    ),

    (
        "Optimized RF model\n"
        "Prediction\n"
        "Model interpretation\n"
        "Chemical motif discovery",
        0.07,
        0.09
    )

]


# =========================
# Draw boxes
# =========================

for text, y, h in boxes:
    add_box(text, y, h)



# =========================
# Draw arrows ONLY between boxes
# =========================

for i in range(len(boxes)-1):

    y1 = boxes[i][1]
    h1 = boxes[i][2]

    y2 = boxes[i+1][1]

    # 上一个框底部
    start = y1

    # 下一个框顶部
    end = y2 + boxes[i+1][2]

    # 留一点空隙
    start -= 0.008
    end += 0.008

    add_arrow(
        start,
        end
    )


# =========================
# Title
# =========================

plt.title(
    "Machine learning framework for DILI risk prediction",
    fontsize=13,
    pad=18
)



plt.xlim(0,1)
plt.ylim(0,1)


plt.tight_layout()



# =========================
# Save
# =========================

plt.savefig(
    "../result/Figure1A_workflow.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()