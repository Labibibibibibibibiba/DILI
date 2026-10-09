import matplotlib.pyplot as plt
from PIL import Image
import os

# =========================
# 修改这里：三个结构图片路径
# =========================

acitretin_path = r"D:\DILI\Acitretin.png"

bexarotene_path = r"D:\DILI\Bexarotene.png"

atomoxetine_path = r"D:\DILI\Atomoxetine Hydrochloride.png"

# 输出路径
output_path = r"D:\DILI\result\Figure3C_representative_compounds.png"

# =========================
# 读取图片
# =========================

imgs = []

for path in [
    acitretin_path,
    bexarotene_path,
    atomoxetine_path
]:

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    img = Image.open(path)

    imgs.append(img)

# =========================
# 创建画布
# =========================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 5)
)

compound_names = [
    "Acitretin",
    "Bexarotene",
    "Atomoxetine"
]

for ax, img, name in zip(
        axes,
        imgs,
        compound_names):
    ax.imshow(img)

    ax.axis("off")

    ax.set_title(
        name,
        fontsize=16,
        fontweight="bold",
        pad=15
    )

# =========================
# 总标题
# =========================

fig.suptitle(
    "Representative DILI-positive compounds containing the enriched chemical motif",
    fontsize=18,
    y=0.98
)

# =========================
# 说明框
# =========================

text = (
    "Highlighted motif:\n"
    "Methyl-substituted aromatic fragment\n"
    "Substructure: cc(c)C"
)

fig.text(
    0.5,
    0.03,
    text,
    ha="center",
    va="center",
    fontsize=12,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black"
    )
)

plt.tight_layout(
    rect=[0, 0.12, 1, 0.93]
)

# =========================
# 保存
# =========================

plt.savefig(
    output_path,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("Figure3C saved:")
print(output_path)