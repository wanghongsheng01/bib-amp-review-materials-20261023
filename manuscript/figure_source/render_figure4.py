#!/usr/bin/env python3
"""Draw Figure 4 from a frozen, qualitative scientific contract.

No source data are downloaded or inferred. The three evaluation endpoints are shown on independent horizontal tracks.
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
QA = Path(os.environ.get("FIGURE4_QA_DIR", ROOT / "qa" / "figure4_v4"))
WIDTH, HEIGHT = 510.0, 252.0
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
QA.mkdir(parents=True, exist_ok=True)
pdf_path = FIG/'fig4_evidence_checked.pdf'
svg_path = FIG/'fig4_evidence_checked.svg'
c = Canvas(str(pdf_path), pagesize=(WIDTH,HEIGHT), pageCompression=1, invariant=1,
           initialFontName='DVS', initialFontSize=MIN_SIZE)
c.setTitle('Figure 4. Three endpoints for evaluating peptide design')
c.setAuthor('BIB AMP manuscript revision')
c.setSubject('Parallel evaluation of sequence reconstruction, generation and antibacterial activity')
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}pt" height="{HEIGHT}pt" viewBox="0 0 {WIDTH} {HEIGHT}">',
       '<title>Three endpoints for evaluating peptide design</title>',
       '<desc>Three independent left-to-right tracks show input, measurement and endpoint. Reconstruction and generation use computational data; antibacterial activity uses the 27-peptide assay panel. There are no cross-track connections.</desc>',
       '<rect x="0" y="0" width="510" height="252" fill="white"/>']
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

# Recompute the displayed values from frozen row-level artifacts.
import csv
from collections import Counter
repo = ROOT.parent
rec = json.loads((repo/'compute/data/manuscripts/review_candidate/evidence/dev_reconstruction_CE.json').read_text())
correct = sum(r['correct_residues'] for r in rec)
positions = sum(r['true_length'] for r in rec)
assert len(rec) == 41 and correct == 497 and positions == 701
samples = json.loads((repo/'compute/data/evidence/amp_h5/feasibility/ce_recovery_four_arm_evaluation_v1/generation_sample_audit.json').read_text())
groups = Counter((r['method'], r['terminal']) for r in samples)
assert len(groups) == 8 and set(groups.values()) == {16}
distances = list(csv.DictReader((repo/'compute/data/manuscripts/review_candidate/evidence/generation_all_distances.csv').open()))
assert len({r['metric'] for r in distances}) == 6
mic = list(csv.DictReader((repo/'wetlab/data/mic_summary_reported.csv').open()))
algorithm = list(csv.DictReader((repo/'wetlab/data/mic_algorithm_calls.csv').open()))
assert len(mic) == 27
sa_count = sum(r['saureus_ug_ml'] != 'N' and float(r['saureus_ug_ml']) <= 31.25 for r in mic)
sa_algorithm = sum(r['strain'] == 'S. aureus ATCC 29213' and bool(r['algorithm_numeric_mic_ug_ml']) and float(r['algorithm_numeric_mic_ug_ml']) <= 31.25 for r in algorithm)
assert sa_count == sa_algorithm == 6
contract = {
    'role': 'endpoint framework; separate computational and laboratory measurements',
    'panels': ['A Reconstruction', 'B Generation', 'C Antibacterial activity'],
    'edges': ['target sequence -> posterior-mean decoding -> residue accuracy',
              'terminal condition -> four generators -> six sequence distances',
              '27-peptide assay panel -> triplicate dilution assays -> strain-specific MIC'],
    'cross_panel_edges': [],
    'values': {'CE21_correct_residues': correct, 'CE21_target_residues': positions,
               'accuracy_percent': round(100*correct/positions, 2), 'development_sequences': len(rec),
               'generator_terminal_groups': len(groups), 'sequences_per_group': 16,
               'assayed_peptides': len(mic), 'sa_Mic_le31_25_reported': sa_count,
               'sa_Mic_le31_25_algorithm': sa_algorithm},
    'forbidden': ['checkpoint-to-peptide attribution', 'purchase/delivery process',
                  'unresolved discrepancy as headline', 'cross-row arrow', 'new data'],
    'baseline_archive': 'revision_v4/baseline/v3.1_tracked_repository.zip',
    'final_width_mm': 178,
    'official_numeric_font_floor': None,
    'local_final_size_target_pt': 9.2,
}
(QA/'figure4-semantic-contract.json').write_text(json.dumps(contract, indent=2)+'\n')

# The left-to-right grammar is local to each endpoint; the three rows are
# parallel scientific questions, not an experimental sequence or model validation.
rows = [
 ('A', 'Reconstruction', 174, BLUE, PALE_BLUE,
  ('Target sequences', '41 development peptides'),
  ('Posterior-mean decoding', 'CE21 versus MSE21'),
  ('Residue accuracy', f'CE21: {correct}/{positions} (70.90%)')),
 ('B', 'Generation', 92, BLUE, PALE_BLUE,
  ('Terminal condition', 'Free / amidated'),
  ('Four generators', '16 sequences per group'),
  ('Six sequence distances', 'LSTM comparator')),
 ('C', 'Antibacterial activity', 10, OCHRE, PALE_OCHRE,
  ('27-peptide assay panel', 'Four bacterial strains'),
  ('Triplicate dilution assays', 'OD600 to MIC'),
  ('S. aureus: 6/27 peptides', 'MIC ≤31.25 μg/mL')),
]
for tag, title, y, color, fill, left, center, right in rows:
    text(tag+'-heading',tag+'  '+title,4,y+58,10.5,True,INK,align='left')
    for j,(x,w,labels) in enumerate([(4,132,left),(161,167,center),(353,153,right)]):
        key=f'{tag}-object-{j}'
        rect(key,x,y,w,43,fill,color,0.65)
        text(key+'-name',labels[0],x+w/2,y+26,9.4,True,owner=key)
        text(key+'-detail',labels[1],x+w/2,y+11,9.4,False,owner=key)
    arrow(tag+'-input-to-measurement',140,157,y+21.5,INK)
    arrow(tag+'-measurement-to-endpoint',332,349,y+21.5,INK)

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
# Poppler rasterization is unavailable in this host; use the verified MuPDF
# renderer for QA only. The submitted PDF and SVG remain vector objects.
from pypdf import PdfReader, PdfWriter
final_scale=(178/25.4*72)/WIDTH
final_page=PdfReader(str(pdf_path)).pages[0]
final_page.scale_by(final_scale)
final_writer=PdfWriter();final_writer.add_page(final_page)
final_pdf=QA/'figure4-final-178mm.pdf'
with final_pdf.open('wb') as handle:final_writer.write(handle)
renderer = os.environ.get('FIGURE_RENDER_PYTHON', '/usr/bin/python3')
render_code = "import fitz,sys; from PIL import Image; d=fitz.open(sys.argv[1]); p=d[0].get_pixmap(dpi=100,alpha=False); p.save(sys.argv[2]); Image.open(sys.argv[2]).convert('L').save(sys.argv[3])"
for source, stem in [(pdf_path,'figure4-100dpi'), (final_pdf,'figure4-final-178mm-100dpi')]:
    subprocess.run([renderer,'-c',render_code,str(source),str(QA/(stem+'-color.png')),str(QA/(stem+'-gray.png'))],check=True)
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
        color_preview=str(QA/'figure4-100dpi-color.png'),grayscale_preview=str(QA/'figure4-100dpi-gray.png'),
        final_color_preview=str(QA/'figure4-final-178mm-100dpi-color.png'),
        final_grayscale_preview=str(QA/'figure4-final-178mm-100dpi-gray.png'),
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pdf_path,svg_path]},
        final_page_check='Required after main.tex includes the figure at full text width.')
(QA/'figure4-verification.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(qa,indent=2,ensure_ascii=False))
