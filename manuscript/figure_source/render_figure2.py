#!/usr/bin/env python3
"""Re-render reported aggregate figures; no experimental recomputation.

Run from any directory with Python, matplotlib and Pillow installed.
Outcomes match Supplementary Table S4. No network calls.
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


def figure2():
    fig, ax = canvas(268)
    text(ax,8,251,"Follow-up route",10.3,"bold")
    text(ax,195,251,"Assessment",10.3,"bold")
    text(ax,365,251,"Observed outcome",10.3,"bold")
    line(ax,8,239,502,239,INK,.8)
    rows=[
      ("Training-posterior draws", "Source diagnostic", "MMD not improved"),
      ("Coordinate permutation", "Source diagnostic", "MMD worsened"),
      ("Reference-template prior", "Source diagnostic", "Mixed; extra information"),
      ("KL warmup", "Reconstruction / resource", "4 guards fail"),
      ("Latent-conditioned AR", "Information / generation", "PASS / FAIL"),
      ("Direct-latent diffusion", "Mechanism / value", "FAIL / not interpreted"),
      ("Conditioning context", "Multistage screen", "Not supported"),
    ]
    for i,(name,assessment,outcome) in enumerate(rows):
      y=223-27*i
      if i%2==0:ax.add_patch(Rectangle((8,y-12),494,24,facecolor=PALE,edgecolor="none"))
      text(ax,12,y,name)
      text(ax,195,y,assessment,size=9.2)
      text(ax,365,y,outcome,size=9.2)
    line(ax,8,46,502,46,INK,.8)
    text(ax,8,31,"First three routes: no applicable generation winner gate.")
    text(ax,8,15,"KL warmup: its separate investment criterion passes.")
    save(fig,ax,"fig2_route_gates")

if __name__ == "__main__":
    figure2()
    (QA / "figure2-geometry-audit.json").write_text(json.dumps(AUDIT,indent=2)+"\n")
