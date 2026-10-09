from PIL import Image, ImageDraw, ImageFont
import os


# ==============================
# 修改这里
# ==============================

structure_dir = r"D:\DILI\data\structures"

output_path = r"D:\DILI\result\Figure3C_final.png"


# 三个结构图片
files = {
    "Acitretin": "Acitretin.png",
    "Bexarotene": "Bexarotene.png",
    "Atomoxetine HCl": "Atomoxetine Hydrochloride.png"
}


# ==============================
# 参数
# ==============================

fig_width = 2400
fig_height = 1400

img_size = (550,550)


# 创建画布

fig = Image.new(
    "RGB",
    (fig_width, fig_height),
    "white"
)

draw = ImageDraw.Draw(fig)


# 字体

try:
    title_font = ImageFont.truetype(
        "arial.ttf",
        80
    )

    name_font = ImageFont.truetype(
        "arial.ttf",
        65
    )

    text_font = ImageFont.truetype(
        "arial.ttf",
        45
    )

except:

    title_font = None
    name_font = None
    text_font = None



# ==============================
# 总标题
# ==============================

title = (
    "Representative DILI-positive compounds "
    "containing the enriched chemical motif"
)

draw.text(
    (fig_width//2,80),
    title,
    anchor="mm",
    font=title_font,
    fill="black"
)



# ==============================
# 放结构
# ==============================

positions = [
    250,
    925,
    1600
]


for i,(name,file) in enumerate(files.items()):

    path=os.path.join(
        structure_dir,
        file
    )


    img=Image.open(path)

    img.thumbnail(img_size)


    x=positions[i]
    y=300


    # 名称

    draw.text(
        (x+275,y-80),
        name,
        anchor="mm",
        font=name_font,
        fill="black"
    )


    # 图片

    fig.paste(
        img,
        (
            x,
            y
        )
    )



# ==============================
# motif说明框
# ==============================


box_x1=650
box_y1=1030
box_x2=1750
box_y2=1280


draw.rounded_rectangle(
    (
        box_x1,
        box_y1,
        box_x2,
        box_y2
    ),
    radius=30,
    outline="black",
    width=5
)


text = (
    "Enriched chemical motif\n\n"
    "Methyl-substituted aromatic fragment\n"
    "FP_367 (cc(c)C)"
)


draw.text(
    (
        (box_x1+box_x2)//2,
        (box_y1+box_y2)//2
    ),
    text,
    anchor="mm",
    align="center",
    font=text_font,
    fill="black"
)



# 保存

fig.save(
    output_path,
    dpi=(600,600)
)


print(
    "Saved:",
    output_path
)