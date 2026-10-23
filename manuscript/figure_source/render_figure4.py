#!/usr/bin/env python3
"""Draw Figure 4 from a frozen, qualitative scientific contract.

No source data are downloaded or inferred. Original summary and retrospective algorithmic counts remain distinct.
PDF and editable SVG are emitted from the same drawing objects. The only Python
dependency for drawing is ReportLab; optional QA requires Pillow, pypdf/pdfplumber
and the local pdftoppm/pdffonts programs.
"""
from pathlib import Path
from dataclasses import dataclass
from html import escape
import json
import hashlib
import os
import shutil
import subprocess

from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
QA = ROOT / 'qa'
WIDTH, HEIGHT = 510.0, 220.0
MIN_SIZE = 9.4
INK = '#242C32'
BLUE = '#25648A'
OCHRE = '#B88426'
RUST = '#AD5C47'
GRAY = '#777777'
PALE_BLUE = '#F0F5F8'
PALE_OCHRE = '#FCF7ED'
PALE_GRAY = '#F7F7F7'
FONT_CANDIDATES = [
    Path(os.environ['FIGURE_FONT_DIR']) if os.environ.get('FIGURE_FONT_DIR') else None,
    Path.home()/'Library/Python/3.9/lib/python/site-packages/matplotlib/mpl-data/fonts/ttf',
    Path('/usr/share/fonts/truetype/dejavu'),
]
font_dir = next((p for p in FONT_CANDIDATES if p and (p/'DejaVuSans.ttf').is_file() and (p/'DejaVuSans-Bold.ttf').is_file()), None)
if font_dir is None:
    raise RuntimeError('DejaVu Sans is required. Set FIGURE_FONT_DIR to its TTF directory.')
pdfmetrics.registerFont(TTFont('DVS', str(font_dir/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DVS-Bold', str(font_dir/'DejaVuSans-Bold.ttf')))
FIG.mkdir(exist_ok=True)
QA.mkdir(exist_ok=True)
pdf_path = FIG/'fig4_evidence_checked.pdf'
svg_path = FIG/'fig4_evidence_checked.svg'
c = Canvas(str(pdf_path), pagesize=(WIDTH,HEIGHT), pageCompression=1, invariant=1,
           initialFontName='DVS', initialFontSize=MIN_SIZE)
c.setTitle('Figure 4. Evidence chain for the reported wet-laboratory record')
c.setAuthor('BIB AMP manuscript revision')
c.setSubject('Reported records, retrospective reconstruction and missing model-to-peptide attribution')
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}pt" height="{HEIGHT}pt" viewBox="0 0 {WIDTH} {HEIGHT}">',
       '<title>Evidence chain for the reported wet-laboratory record</title>',
       '<desc>Numbers map to Box 1. A broken model-to-peptide link is distinct from reported assay records and retrospective analysis. Computational-development disclosure is shown separately.</desc>',
       '<rect x="0" y="0" width="510" height="220" fill="white"/>']
texts, shapes, connectors = [], [], []

def rect(key,x,y,w,h,fill,stroke,sw=0.7):
    c.setFillColor(HexColor(fill)); c.setStrokeColor(HexColor(stroke)); c.setLineWidth(sw)
    c.rect(x,y,w,h,stroke=1,fill=1)
    svg.append(f'<rect id="{key}" x="{x}" y="{HEIGHT-y-h}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    shapes.append(dict(id=key,x=x,y=y,w=w,h=h))

def text(key,value,x,y,size=MIN_SIZE,bold=False,color=INK,align='center',owner=None):
    assert size >= MIN_SIZE
    font = 'DVS-Bold' if bold else 'DVS'
    width = pdfmetrics.stringWidth(value,font,size)
    left = x-width/2 if align=='center' else x
    c.setFillColor(HexColor(color)); c.setFont(font,size)
    c.drawString(left,y,value)
    anchor = 'middle' if align=='center' else 'start'
    weight = '700' if bold else '400'
    svg.append(f'<text id="{key}" x="{x}" y="{HEIGHT-y}" text-anchor="{anchor}" font-family="DejaVu Sans" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(value)}</text>')
    asc = pdfmetrics.getAscent(font)*size/1000
    desc = pdfmetrics.getDescent(font)*size/1000
    texts.append(dict(id=key,text=value,x=left,y=y+desc,width=width,height=asc-desc,font=font,size_pt=size,owner=owner))

def line(key,x1,y1,x2,y2,color=GRAY,sw=0.7,dash=None):
    c.setStrokeColor(HexColor(color)); c.setLineWidth(sw); c.setDash(dash or [])
    c.line(x1,y1,x2,y2); c.setDash([])
    dash_attr = f' stroke-dasharray="{",".join(map(str,dash))}"' if dash else ''
    svg.append(f'<path id="{key}" d="M {x1},{HEIGHT-y1} L {x2},{HEIGHT-y2}" fill="none" stroke="{color}" stroke-width="{sw}"{dash_attr}/>')
    connectors.append(dict(id=key,x1=x1,y1=y1,x2=x2,y2=y2,type='line'))

def arrow(key,x1,x2,y,color=INK):
    shaft, head = 0.48, 3.4
    points = [(x1,y-shaft),(x2-head,y-shaft),(x2-head,y-2.3),(x2,y),(x2-head,y+2.3),(x2-head,y+shaft),(x1,y+shaft)]
    c.setFillColor(HexColor(color)); p=c.beginPath(); p.moveTo(*points[0])
    for pt in points[1:]:p.lineTo(*pt)
    p.close();c.drawPath(p,fill=1,stroke=0)
    svg.append(f'<polygon id="{key}" points="'+ ' '.join(f'{x},{HEIGHT-yy}' for x,yy in points)+f'" fill="{color}"/>')
    connectors.append(dict(id=key,x1=x1,y1=y,x2=x2,y2=y,type='arrow'))

contract=json.loads((ROOT/'figure_source/data-contract.json').read_text())
assert contract['od600_count']==3240 and contract['agreement_legacy_N']==105
sa=next(x for x in contract['strains'] if x['key']=='saureus')
assert sa['reported']['below200']==6 and sa['algorithm']['below200']==7

# Main object spine. Run fields are requirements, not known model assignments.
rect('required-run',4,90,71,62,PALE_GRAY,GRAY)
text('run-label-1','Model run',39.5,140,9.7,True,owner='required-run')
text('run-label-2','identifiers',39.5,128,9.7,True,owner='required-run')
text('run-fields-1','checkpoint',39.5,116,owner='required-run')
text('run-fields-2','config. / seed',39.5,104,owner='required-run')

# Missing per-peptide binding: no arrowhead and a real interruption.
text('missing-label-1','2 No run',106,143,9.7,True,RUST)
text('missing-label-2','assignment',106,131,9.4,False,RUST)
line('missing-left',75,117,97,117,RUST,0.9,[3,2])
line('missing-right',115,117,137,117,RUST,0.9,[3,2])
line('break-1',101,113,104,121,RUST,0.9)
line('break-2',108,113,111,121,RUST,0.9)

rect('peptide-record',137,77,84,75,PALE_BLUE,BLUE)
text('peptide-label','Peptide IDs',179,137,9.7,True,owner='peptide-record')
text('ordered-count','30 ordered',179,120,owner='peptide-record')
text('assayed-count','27 assayed',179,107,owner='peptide-record')
text('mapped-ids','IDs mapped',179,94,owner='peptide-record')

# Box 1 item 1 is contextual, not a certified prospective operation.
rect('selection-record',108,171,142,43,PALE_BLUE,BLUE)
text('selection-label','1 Candidate selection',179,201,9.7,True,owner='selection-record')
text('selection-description','Historical screening rules',179,189,owner='selection-record')
text('selection-time','30 sequences ordered',179,177,owner='selection-record')
line('selection-context',179,171,179,152,BLUE,0.7)

arrow('peptide-to-readout',222,236,121)
rect('raw-readings',237,77,82,75,PALE_BLUE,BLUE)
text('readout-label','5 Assay data',278,141,9.7,True,owner='raw-readings')
text('readout-type-1','3,240 OD600',278,128,owner='raw-readings')
text('readout-type-2','values + 108',278,115,owner='raw-readings')
text('readout-provenance-1','MIC cells',278,102,owner='raw-readings')


# Retrospective check and original summary remain distinct representations.
# Located discrepancies are not adjudicated MIC corrections.
line('assay-to-branch',319,121,332,121,INK,0.95)
line('branch-vertical',332,92,332,173,INK,0.95)
arrow('assay-to-mic-check',332,348,173)
arrow('assay-to-reported-count',332,348,92)
rect('mic-reconstruction',350,142,156,60,PALE_OCHRE,OCHRE)
text('mic-label','4 Endpoint comparison',428,190,9.7,True,owner='mic-reconstruction')
text('mic-state','105/108 summary matches',428,177,owner='mic-reconstruction')
text('agreement-count','3 differing endpoints',428,164,owner='mic-reconstruction')
text('unresolved-count','Retrospective rule',428,151,owner='mic-reconstruction')

rect('activity-count',350,59,156,67,PALE_BLUE,BLUE)
text('activity-label','3 S. aureus record',428,114,9.7,True,owner='activity-count')
text('activity-rates','MIC <200 μg/mL',428,102,owner='activity-count')
text('activity-criterion','Reported: 6/27 (6/30)',428,90,owner='activity-count')
text('activity-time','Algorithm: 7/27 (7/30)',428,78,owner='activity-count')
text('activity-impact','assayed (ordered)',428,66,owner='activity-count')

# Computational-development evidence is separate; there is deliberately no
# connector linking it to the wet-laboratory chain or model-origin requirement.
line('separate-record-rule',4,48,506,48,'#C4C9CD',0.55)
text('development-header','6 Computational comparison (separate)',4,32,9.7,True,align='left')
text('development-state','279 training and 41 development sequences; one fitted seed; four generators.',4,16,9.4,False,INK,align='left')

svg.append('</svg>')
svg_path.write_text('\n'.join(svg)+'\n',encoding='utf-8')
c.showPage();c.save()

# Geometry verification uses actual embedded-font metrics, not label lengths.
errors=[]
shape_map={s['id']:s for s in shapes}
for t in texts:
    if t['x'] < 0 or t['x']+t['width'] > WIDTH or t['y'] < 0 or t['y']+t['height'] > HEIGHT:
        errors.append('Outside canvas: '+t['id'])
    if t['owner']:
        s=shape_map[t['owner']]
        if t['x'] < s['x']+2 or t['x']+t['width'] > s['x']+s['w']-2 or t['y'] < s['y']+2 or t['y']+t['height'] > s['y']+s['h']-2:
            errors.append('Insufficient owner clearance: '+t['id'])
def overlaps(a,b):
    return a['x'] < b['x']+b['width'] and a['x']+a['width'] > b['x'] and a['y'] < b['y']+b['height'] and a['y']+a['height'] > b['y']
for i,a in enumerate(texts):
    for b in texts[i+1:]:
        if overlaps(a,b):errors.append('Text intersection: '+a['id']+' / '+b['id'])
for e in connectors:
    bb=dict(x=min(e['x1'],e['x2'])-0.6,y=min(e['y1'],e['y2'])-0.6,width=abs(e['x1']-e['x2'])+1.2,height=abs(e['y1']-e['y2'])+1.2)
    for t in texts:
        if overlaps(bb,t):errors.append('Connector/text intersection: '+e['id']+' / '+t['id'])
inventory=dict(canvas_pt=[WIDTH,HEIGHT],font_directory=str(font_dir),minimum_font_pt=min(t['size_pt'] for t in texts),texts=texts,shapes=shapes,connectors=connectors,geometry_errors=errors)
(QA/'figure4-labels-and-geometry.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False)+'\n')
if errors:raise RuntimeError('; '.join(errors))

pdf_font_report=subprocess.run(['pdffonts',str(pdf_path)],capture_output=True,text=True,check=True).stdout
(QA/'figure4-fonts.txt').write_text(pdf_font_report)
subprocess.run(['pdftoppm','-r','100','-png','-singlefile',str(pdf_path),str(QA/'figure4-100dpi-color')],check=True)
from PIL import Image
with Image.open(QA/'figure4-100dpi-color.png') as im:
    im.convert('L').save(QA/'figure4-100dpi-gray.png')
from pypdf import PdfReader, PdfWriter
final_scale=(178/25.4*72)/WIDTH
final_page=PdfReader(str(pdf_path)).pages[0]
final_page.scale_by(final_scale)
final_writer=PdfWriter();final_writer.add_page(final_page)
final_pdf=QA/'figure4-final-178mm.pdf'
with final_pdf.open('wb') as handle:final_writer.write(handle)
subprocess.run(['pdftoppm','-r','300','-png','-singlefile',str(final_pdf),str(FIG/'fig4_evidence_checked')],check=True)
subprocess.run(['pdftoppm','-r','100','-png','-singlefile',str(final_pdf),str(QA/'figure4-final-178mm-100dpi-color')],check=True)
with Image.open(QA/'figure4-final-178mm-100dpi-color.png') as im:
    im.convert('L').save(QA/'figure4-final-178mm-100dpi-gray.png')
import pdfplumber
with pdfplumber.open(pdf_path) as doc:
    p=doc.pages[0]
    pdf_min=min(ch['size'] for ch in p.chars)
    actual_text=p.extract_text()
    fonts=sorted(set(ch['fontname'] for ch in p.chars))
    dimensions=[p.width,p.height]
    if pdf_min < MIN_SIZE-0.001:raise RuntimeError('PDF glyph size below contract')
with pdfplumber.open(final_pdf) as doc:
    final_pdf_min=min(ch['size'] for ch in doc.pages[0].chars)
    if final_pdf_min < 9.2:raise RuntimeError('Actual 178 mm glyph size below 9.2 pt')
(QA/'figure4-pdf-text.txt').write_text(actual_text+'\n')
qa=dict(canvas_pt=dimensions,actual_pdf_minimum_font_pt=pdf_min,final_width_mm=178,
        final_minimum_font_pt_at_178mm=final_pdf_min,embedded_font_names=fonts,
        label_count=len(texts),svg_editable_text_count=len(texts),geometry_error_count=len(errors),
        color_preview='qa/figure4-100dpi-color.png',grayscale_preview='qa/figure4-100dpi-gray.png',
        final_color_preview='qa/figure4-final-178mm-100dpi-color.png',
        final_grayscale_preview='qa/figure4-final-178mm-100dpi-gray.png',
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pdf_path,svg_path]},
        final_page_check='Required after main.tex includes the figure at full text width.')
(QA/'figure4-verification.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(qa,indent=2,ensure_ascii=False))
