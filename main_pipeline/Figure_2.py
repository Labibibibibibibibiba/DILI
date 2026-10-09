# ============================================================
# Figure 2 assembly
# SHAP interpretation figure
# ============================================================


from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# ============================================================
# 修改路径
# ============================================================


panel_A = r"D:\DILI\result\SHAP_summary_plot.png"

panel_B = r"D:\DILI\result\Figure2B_SHAP_feature_importance.png"

panel_C = r"D:\DILI\gpt1_Acitretin.png"


output = r"D:\DILI\result\Figure2_SHAP_interpretation.png"



# ============================================================
# 读取图片
# ============================================================


imgA = Image.open(panel_A)
imgB = Image.open(panel_B)
imgC = Image.open(panel_C)



print("A:", imgA.size)
print("B:", imgB.size)
print("C:", imgC.size)



# ============================================================
# resize函数
# ============================================================


def resize_keep(img, width):

    ratio = width / img.width

    height = int(img.height * ratio)

    return img.resize(
        (width,height)
    )



# ============================================================
# 调整大小
# ============================================================


A = resize_keep(imgA,1200)

B = resize_keep(imgB,600)

C = resize_keep(imgC,600)



# ============================================================
# Canvas
# ============================================================


margin = 80


canvas_width = 1400


canvas_height = (
    A.height
    +
    max(B.height,C.height)
    +
    margin*3
)


canvas = Image.new(
    "RGB",
    (
        canvas_width,
        canvas_height
    ),
    "white"
)



# ============================================================
# 字体
# ============================================================


try:

    font_big = ImageFont.truetype(
        "arial.ttf",
        80
    )

except:

    font_big = None



# ============================================================
# Panel A
# ============================================================


xA = (canvas_width-A.width)//2

yA = margin


canvas.paste(
    A,
    (
        xA,
        yA
    )
)



# label A

draw = ImageDraw.Draw(canvas)

draw.text(
    (
        30,
        yA
    ),
    "A",
    font=font_big,
    fill="black"
)



# ============================================================
# Panel B/C
# ============================================================


y_bottom = A.height + margin*2


# B

canvas.paste(
    B,
    (
        80,
        y_bottom
    )
)


draw.text(
    (
        50,
        y_bottom
    ),
    "B",
    font=font_big,
    fill="black"
)



# C

canvas.paste(
    C,
    (
        760,
        y_bottom
    )
)


draw.text(
    (
        730,
        y_bottom
    ),
    "C",
    font=font_big,
    fill="black"
)



# ============================================================
# 保存
# ============================================================


canvas.save(
    output,
    dpi=(600,600)
)


print("================")
print("Figure 2 finished")
print(output)
print("================")


plt.figure(figsize=(12,12))
plt.imshow(canvas)
plt.axis("off")
plt.show()