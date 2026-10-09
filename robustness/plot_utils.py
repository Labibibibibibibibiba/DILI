#!/usr/bin/env python3
"""
plot_utils.py — DILI 论文统一绘图底座

融合两个 skill 的规范：
- scientific-visualization：出版级样式、诚实编码、面板标注、期刊尺寸、字体嵌入
- data-visualization：Okabe-Ito 色盲安全配色（Nature 推荐）、300 DPI 最低、安全图尺寸

所有 4 张 figure 共用本模块，保证风格统一、可复现、色盲安全。
使用方式：

    import plot_utils as pu
    pu.apply_paper_style()
    fig, ax = plt.subplots(figsize=pu.figsize(pu.SINGLE_COL_MM, aspect=0.75),
                           layout="constrained")
    ...
    pu.label_panel(ax, "A")
    pu.save_figure(fig, "outputs/fig1", formats=("pdf", "png"), dpi=600)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Iterable, Optional

# =============================================================================
# 1. 期刊尺寸（Nature flagship 单栏/双栏，mm）
# =============================================================================
SINGLE_COL_MM = 89.0       # 单栏
ONE_HALF_MIN_MM = 120.0    # 1.5 栏（最小）
DOUBLE_COL_MM = 183.0      # 双栏/全宽
MAX_HEIGHT_MM = 247.0      # 最大高度


def mm2inch(mm: float) -> float:
    """毫米转英寸。"""
    return mm / 25.4


def figsize(width_mm: float = SINGLE_COL_MM, height_mm: Optional[float] = None,
            aspect: float = 0.75) -> tuple[float, float]:
    """返回 (width_inch, height_inch)，默认 4:3 宽高比。"""
    w = mm2inch(width_mm)
    h = mm2inch(height_mm) if height_mm is not None else w * aspect
    return (w, h)


# =============================================================================
# 2. Okabe-Ito 色盲安全配色（Nature Methods 推荐，色盲友好）
# =============================================================================
OKABE_ITO = {
    "orange":          "#E69F00",
    "sky_blue":        "#56B4E9",
    "bluish_green":    "#009E73",
    "yellow":          "#F0E442",
    "blue":            "#0072B2",
    "vermillion":      "#D55E00",
    "reddish_purple":  "#CC79A7",
    "black":           "#000000",
}

# 模型对比（Fig1C）—— RF 是最终选定模型，用最深蓝突出；LR 用浅蓝；XGB 用橙
MODEL_COLORS = {
    "Logistic Regression": "#56B4E9",   # sky blue（浅，非主模型）
    "Random Forest":       "#0072B2",   # blue（深，主模型，突出）
    "XGBoost":             "#E69F00",   # orange
}

# 药物分组（Fig1D / Fig3D）—— 语义：橙红=阳性对照（警示），蓝=高风险候选，蓝绿=阴性对照
DRUG_GROUP_COLORS = {
    "positive_control":     "#D55E00",  # vermillion 阳性对照
    "high_risk_candidate":  "#0072B2",  # blue 高风险候选
    "negative_control":     "#009E73",  # bluish green 阴性对照
}

# 药物分组标记形状（冗余编码，不单靠颜色区分）
DRUG_GROUP_MARKERS = {
    "positive_control":     "o",        # circle
    "high_risk_candidate":  "s",        # square
    "negative_control":     "^",        # triangle
}

# 具体候选药 → 分组颜色（Fig3D 直接查表）
COMPOUND_GROUPS = {
    "Chlorpromazine": "positive_control",
    "Diclofenac":     "positive_control",
    "Sulfamethizole": "high_risk_candidate",
    "Cabotegravir":   "high_risk_candidate",
    "Mebendazole":    "high_risk_candidate",
    "Oxybutynin":     "negative_control",
    "Tolterodine":    "negative_control",
    "Metformin":      "negative_control",
}

# 不确定类别（Fig3A / Fig3B）
CATEGORY_COLORS = {
    "vLess-DILI-concern":     "#0072B2",  # blue
    "Ambiguous-DILI-concern": "#E69F00",  # orange
}

# DILIrank 2.0 四分类（Fig1B）—— 语义色：红=高风险，浅蓝=低风险，蓝绿=安全，橙=不确定
DILI_CLASS_COLORS = {
    "vMost-DILI-concern":     "#D55E00",  # vermillion 高风险
    "vLess-DILI-concern":     "#56B4E9",  # sky blue 低风险
    "vNo-DILI-concern":       "#009E73",  # bluish green 安全
    "Ambiguous-DILI-concern": "#E69F00",  # orange 不确定
}

# 连续风险谱（Fig3B 热力/尾部高亮）—— 蓝(低)→黄→橙红(高)
RISK_CMAP = "viridis"

# =============================================================================
# 3. 出版级全局样式（scientific-visualization BASE_STYLE 精简）
# =============================================================================
def apply_paper_style() -> None:
    """应用出版级全局样式：无衬线字体、7-8pt、Type 42 字体嵌入、300 DPI。"""
    mpl.rcParams.update({
        # figure
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": False,   # 用显式 layout="constrained"
        # font
        "font.size": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        # axes
        "axes.linewidth": 0.6,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.labelweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.grid": False,
        # ticks
        "xtick.major.size": 3,
        "xtick.major.width": 0.6,
        "xtick.labelsize": 7,
        "xtick.direction": "out",
        "ytick.major.size": 3,
        "ytick.major.width": 0.6,
        "ytick.labelsize": 7,
        "ytick.direction": "out",
        # lines
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.6,
        # legend
        "legend.fontsize": 7,
        "legend.frameon": False,
        # savefig
        "savefig.dpi": 300,
        "savefig.bbox": "standard",   # 精确物理尺寸，不用 tight（除非显式）
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        # image
        "image.cmap": "viridis",
        # 字体嵌入（Type 42 = TrueType 可编辑/嵌入 PDF/PS）
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# =============================================================================
# 4. 面板标注（A/B/C/D）
# =============================================================================
def label_panel(ax, label: str, x: float = -0.12, y: float = 1.05,
                fontsize: float = 8.0, weight: str = "bold") -> None:
    """在面板左上角标注 A/B/C/D（粗体，位于 axes 外）。"""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=weight,
            va="top", ha="left")


# =============================================================================
# 5. 显著性标注（诚实编码：标检验名，不用星号代替）
# =============================================================================
def add_significance(ax, x1: float, x2: float, y: float, text: str,
                     dy: float = 0.02, fontsize: float = 7.0,
                     color: str = "black") -> None:
    """在两组之间画显著性横线 + 文本（如 "P = 0.118"）。

    y 为横线高度（数据坐标），dy 为文本相对横线的抬高量（数据坐标比例）。
    """
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    line_y = y
    text_y = y + dy * span
    ax.plot([x1, x1, x2, x2], [line_y, line_y + 0.5 * dy * span,
                               line_y + 0.5 * dy * span, line_y],
            lw=0.8, color=color, clip_on=False)
    ax.text((x1 + x2) / 2, text_y, text, ha="center", va="bottom",
            fontsize=fontsize, color=color)


# =============================================================================
# 6. 统一导出（PNG + PDF + TIFF，300+ DPI，白色背景）
# =============================================================================
def save_figure(fig, path, formats: Iterable[str] = ("pdf", "png"),
                dpi: int = 300, tight: bool = False,
                facecolor: str = "white") -> list[Path]:
    """统一导出到多格式。

    - 默认 PDF（矢量，投稿）+ PNG（预览），可加 "tiff"。
    - tight=False 保留精确物理尺寸（scientific-visualization 原则：
      bbox_inches="tight" 会改变页面物理尺寸，仅在需要裁边时用）。
    - 300 DPI 是投稿最低标准；线图/组合图建议 600。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        kw = dict(dpi=dpi, facecolor=facecolor, edgecolor="none")
        if tight:
            kw["bbox_inches"] = "tight"
        fig.savefig(out, **kw)
        saved.append(out)
    print(f"Saved: {path.name} [{', '.join(formats)}] @ {dpi} dpi")
    return saved


# =============================================================================
# 7. 组合图导出（scientific-visualization 的 provenance 轻量版）
# =============================================================================
def export_figure(fig, path, formats: Iterable[str] = ("pdf", "png"),
                  dpi: int = 600, provenance: Optional[dict] = None) -> list[Path]:
    """导出并可选附带 provenance 清单（诚实记录数据来源与变换）。"""
    saved = save_figure(fig, path, formats=formats, dpi=dpi, tight=False)
    if provenance is not None:
        import json
        meta_path = Path(path).with_suffix(".provenance.json")
        meta_path.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Saved: {meta_path.name}")
    return saved
