#!/usr/bin/env python3
"""Render Figure 1 from the independently verified saved-output summaries.

Run from any directory with Python, matplotlib and Pillow installed.
Values match compute/results/S1* and S2* tables. No network calls.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
QA = ROOT / "qa"
for folder in (FIG, QA):
    folder.mkdir(exist_ok=True)

BLUE = "#25648A"
RUST = "#AD5C47"
OCHRE = "#B88426"
GRAY = "#777777"
INK = "#222222"
RULE = "#C7CBCF"
PALE = "#F1F3F4"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "bib-amp-2026-10-22-v2", "axes.unicode_minus": True,
})
AUDIT = []


def canvas(height):
    fig = plt.figure(figsize=(510 / 72, height / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 510)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def text(ax, x, y, value, size=9.4, weight="normal", color=INK,
         ha="left", va="center", **kwargs):
    return ax.text(x, y, value, fontsize=size, fontweight=weight,
                   color=color, ha=ha, va=va, **kwargs)


def line(ax, x1, y1, x2, y2, color=RULE, lw=0.65):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="butt")


def bar(ax, x, y, width, height, color):
    ax.add_patch(Rectangle((x, y-height/2), width, height,
                          facecolor=color, edgecolor=color, linewidth=0.6))


def axis(ax, x, y, width, xmax, ticks, labels):
    line(ax, x, y, x + width, y, INK, 0.65)
    for tick, label in zip(ticks, labels):
        px = x + width * tick / xmax
        line(ax, px, y, px, y-3, INK, 0.65)
        text(ax, px, y-11, label, ha="center")


def save(fig, ax, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for t in ax.texts:
        b = t.get_window_extent(renderer)
        texts.append({"text": t.get_text(), "size_pt": t.get_fontsize(),
                      "bbox_pt": [round(v, 3) for v in b.extents]})
    width, height = fig.get_size_inches()*72
    outside = [t for t in texts if t["bbox_pt"][0] < 0.5 or
               t["bbox_pt"][1] < 0.5 or t["bbox_pt"][2] > width-0.5 or
               t["bbox_pt"][3] > height-0.5]
    overlaps = []
    for i, a in enumerate(texts):
        x0,y0,x1,y1=a["bbox_pt"]
        for b in texts[i+1:]:
            u0,v0,u1,v1=b["bbox_pt"]
            if min(x1,u1)>max(x0,u0) and min(y1,v1)>max(y0,v0):
                overlaps.append([a["text"],b["text"]])
    if outside or overlaps:
        raise RuntimeError(json.dumps({"figure":stem,"outside":outside,
                                       "text_overlaps":overlaps},indent=2))
    fig.savefig(FIG / f"{stem}.pdf", metadata={
        "Title": stem, "Author": "BIB AMP manuscript revision",
        "Subject": "Reported aggregate evidence, not a new experiment",
        "Creator": "scripts/render_figures1_3.py", "CreationDate": None,
        "ModDate": None})
    fig.savefig(FIG / f"{stem}.svg", metadata={"Date": None})
    fig.savefig(QA / f"{stem}-100dpi.png", dpi=100)
    with Image.open(QA / f"{stem}-100dpi.png") as im:
        im.convert("L").save(QA / f"{stem}-100dpi-gray.png")
    # The native 510 pt canvas is scaled to the template's actual 178 mm width.
    # This optional PDF rasterization is the standalone final-size preflight.
    if shutil.which("pdftoppm"):
        target = QA / f"{stem}-final178mm-100dpi"
        subprocess.run(["pdftoppm", "-singlefile", "-png", "-r", "100",
                        "-scale-to-x", str(round(178 / 25.4 * 100)),
                        "-scale-to-y", "-1", str(FIG / f"{stem}.pdf"),
                        str(target)], check=True, capture_output=True)
        with Image.open(target.with_suffix(".png")) as im:
            im.convert("L").save(QA / f"{stem}-final178mm-100dpi-gray.png")
    audit = {"figure": stem, "width_pt": width, "height_pt": height,
             "minimum_explicit_font_pt": min(t["size_pt"] for t in texts),
             "out_of_canvas_text": outside, "text_overlaps": overlaps,
             "texts": texts, "sha256": {}}
    for ext in ("pdf","svg"):
        p=FIG / f"{stem}.{ext}"
        audit["sha256"][ext]=hashlib.sha256(p.read_bytes()).hexdigest()
    AUDIT.append(audit)
    plt.close(fig)


def figure1():
    fig, ax = canvas(200)
    panels = [
        (0, "A  Reconstruction", ["CE21","MSE21","scalar"],
         [70.90,12.55,15.83],110,[0,50,100],["0","50","100"],
         "Residue micro accuracy (%)", [BLUE,RUST,GRAY]),
        (171,"B  Free terminal",["CE21","MSE21","scalar","LSTM"],
         [0.0173,0.0823,0.0399,0.0116],.125,[0,.04,.08],["0",".04",".08"],
         "Development MMD²",[BLUE,RUST,GRAY,OCHRE]),
        (342,"C  Amidated terminal",["CE21","MSE21","scalar","LSTM"],
         [0.0077,0.0848,0.0313,0.0074],.125,[0,.04,.08],["0",".04",".08"],
         "Development MMD²",[BLUE,RUST,GRAY,OCHRE]),
    ]
    for i, (left,title,names,values,xmax,ticks,ticklabels,xlabel,colors) in enumerate(panels):
        text(ax,left+2,188,title,10.3,"bold")
        x=left+39
        for j,(name,value,color) in enumerate(zip(names,values,colors)):
            y=154-j*27
            text(ax,x-5,y,name,ha="right")
            width=118*value/xmax
            bar(ax,x,y,width,15,color)
            text(ax,x+width+3,y,f"{value:.2f}" if i==0 else f"{value:.4f}")
        axis(ax,x,48,118,xmax,ticks,ticklabels)
        text(ax,left+83,21,xlabel,ha="center")
        text(ax,left+83,8,"Higher is better" if i==0 else "Lower is closer",ha="center")
    # A's paired contrast is printed without inventing an uncertainty interval.
    text(ax,4,69,"CE21 − MSE21: +58.35 pp")
    save(fig,ax,"fig1_reconstruction_vs_generation")


if __name__ == "__main__":
    figure1()
