#!/usr/bin/env python3
"""Correct Figure 3 factual values; original and retrospective algorithm are distinct.

Run from any directory with Python, matplotlib and Pillow installed.
All values are frozen in data-contract.json. No network calls.
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
        "Subject": "Original summary and retrospective algorithm; not adjudicated MIC",
        "Creator": "render_fig3_checked.py", "CreationDate": None,
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
        subprocess.run(["pdftoppm", "-singlefile", "-png", "-r", "300",
                        "-scale-to-x", str(round(178 / 25.4 * 300)),
                        "-scale-to-y", "-1", str(FIG / f"{stem}.pdf"),
                        str(FIG / stem)], check=True, capture_output=True)
    audit = {"figure": stem, "width_pt": width, "height_pt": height,
             "minimum_explicit_font_pt": min(t["size_pt"] for t in texts),
             "out_of_canvas_text": outside, "text_overlaps": overlaps,
             "texts": texts, "sha256": {}}
    for ext in ("pdf","svg"):
        p=FIG / f"{stem}.{ext}"
        audit["sha256"][ext]=hashlib.sha256(p.read_bytes()).hexdigest()
    AUDIT.append(audit)
    plt.close(fig)


def source_bar(ax,x,y,width,height,color,reported):
    if width<=0:return
    ax.add_patch(Rectangle((x,y-height/2),width,height,
        facecolor=color if reported else 'white',edgecolor=color,linewidth=.8))

contract=json.loads((ROOT/'figure_source/data-contract.json').read_text())
fig,ax=canvas(322)
ax.set_ylim(-20,302)
text(ax,4,288,'A  Per-strain counts',10.3,'bold')
text(ax,278,288,'B  S. aureus record',10.3,'bold')
bar(ax,7,266,10,7,BLUE);text(ax,22,266,'Numeric MIC')
bar(ax,114,266,10,7,OCHRE);text(ax,129,266,'MIC <32 μg/mL')
text(ax,278,266,'MIC in μg/mL')
source_bar(ax,7,246,10,7,GRAY,True);text(ax,22,246,'Original reported summary')
source_bar(ax,278,246,10,7,GRAY,False);text(ax,293,246,'Retrospective algorithm')
x=84;width=145;xmax=13
for j,r in enumerate(contract['strains']):
    y=211-j*46
    text(ax,x-5,y,r['strain'],ha='right',fontstyle='italic')
    for key,offset,color in [('numeric',15,BLUE),('below32',-7,OCHRE)]:
        for source,delta,solid in [('reported',0,True),('algorithm',-10,False)]:
            yy=y+offset+delta;v=r[source][key];w=width*v/xmax
            source_bar(ax,x,yy,w,7,color,solid);text(ax,x+w+3,yy,str(v))
axis(ax,x,40,width,xmax,[0,4,8,12],['0','4','8','12'])
text(ax,145,15,'Peptides (of 27)',ha='center')
x=366;width=131;xmax=18
for j,r in enumerate(contract['saureus_bins_ug_ml']):
    y=211-j*46
    text(ax,x-6,y,('N code' if r['label']=='N' else 'MIC '+r['label']),ha='right')
    color=GRAY if r['label']=='N' else BLUE if j==0 else OCHRE
    for source,delta,solid in [('reported',5,True),('algorithm',-5,False)]:
        v=r[source];w=width*v/xmax
        source_bar(ax,x,y+delta,w,7,color,solid);text(ax,x+w+3,y+delta,str(v))
axis(ax,x,40,width,xmax,[0,8,16],['0','8','16'])
text(ax,432,15,'Peptides (of 27)',ha='center')
text(ax,4,-10,'S. aureus <200 μg/mL: reported 6/27; retrospective algorithm 7/27.')
save(fig,ax,'fig3_mic_checked')
(QA/'fig3-geometry-audit.json').write_text(json.dumps(AUDIT,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'pass','figure':'fig3_mic_checked','data_contract':'data-contract.json'}))

