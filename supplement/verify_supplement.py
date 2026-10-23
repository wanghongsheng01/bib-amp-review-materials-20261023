#!/usr/bin/env python3
"""Verify complete supplementary tables offline, including row-level arithmetic."""
from pathlib import Path
from collections import defaultdict,Counter
import json,csv,hashlib,math
R=Path(__file__).resolve().parent;B=R.parent;T=R/'tables'
manifest=json.loads((R/'table-provenance.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
def resolve(p):
    candidates=[B/p,B/p.replace('computational/reviewer_candidate/','compute/').replace('wetlab/public_candidate/','wetlab/'),B/'reviewer_repository'/p]
    q=next((x for x in candidates if x.is_file()),None)
    if q is None:raise FileNotFoundError(p)
    return q
checks=0
for x in manifest['tables']:
    p=R/x['output'];assert sha(p)==x['sha256'];assert len(rows(p))==x['rows'];checks+=2
    assert sha(resolve(x['source']))==x['source_sha256'];checks+=1
    for s in x.get('additional_sources',[]):assert sha(resolve(s['path']))==s['sha256'];checks+=1
recs=rows(T/'Table_S1b_reconstruction_per_peptide.csv');assert len(recs)==960
for x in recs:
    t,p=x['truth_sequence'],x['prediction_sequence'];n=sum(i<len(p) and a==p[i] for i,a in enumerate(t))
    assert n==int(x['correct_target_positions']) and abs(len(t)-len(p))==int(x['length_absolute_error'])
    assert len(t)==int(x['target_length']) and len(p)==int(x['prediction_length'])
    assert math.isclose(n/len(t),float(x['aligned_accuracy']),rel_tol=1e-12)
    assert (t==p)==(x['exact_sequence']=='True');checks+=4
od=rows(T/'Table_S5d_OD600_readings.csv');assert len(od)==3240
rep=defaultdict(list)
for x in od:rep[x['sample_id'],x['strain'],x['replicate']].append((float(x['concentration_ug_ml']),float(x['od600'])))
assert len(rep)==324
endpoints={}
for key,v in rep.items():
    assert len(v)==10 and {c for c,o in v}=={1000/2**j for j in range(10)}
    a=[c for c,o in v if o<.1];endpoints[key]=min(a) if a else 'N';checks+=1
names={'E. coli ATCC 25922':'ecoli','P. aeruginosa ATCC 27853':'paeruginosa','S. aureus ATCC 29213':'saureus','E. faecalis ATCC 29212':'efaecalis'}
def v(x):return x if x=='N' else float(x)
mic=rows(T/'Table_S5b_MIC_reported_algorithm.csv');diff=[]
for x in mic:
    r=[endpoints[x['sample_id'],names[x['strain']],str(i)] for i in [1,2,3]]
    for i,z in enumerate(r,1):assert z==v(x['replicate_'+str(i)])
    z,n=Counter(r).most_common(1)[0];z=z if n>=2 else 'N'
    assert z==v(x['legacy_algorithm_N_code'])
    status='no_consensus' if n<2 else 'no_endpoint_majority' if z=='N' else 'numeric_consensus'
    assert status==x['algorithm_status'];checks+=5
    if z!=v(x['reported_summary']):diff.append(x)
assert len(mic)==108 and len(diff)==3
assert diff==rows(T/'Table_S5c_MIC_discrepancies.csv')
sa=[x for x in mic if names[x['strain']]=='saureus']
assert sum(v(x['reported_summary'])!='N' and v(x['reported_summary'])<200 for x in sa)==6
assert sum(v(x['legacy_algorithm_N_code'])!='N' and v(x['legacy_algorithm_N_code'])<200 for x in sa)==7
assert all(sum(v(x[k])!='N' and v(x[k])<32 for x in sa)==6 for k in ['reported_summary','legacy_algorithm_N_code'])
architecture=json.loads((R/'source/architecture_facts.json').read_text())
references={}
def collect(v):
    if isinstance(v,dict):
        if 'repository_relative_path' in v:
            path=v['repository_relative_path'];assert path not in references or references[path]==v['sha256'];references[path]=v['sha256']
        for z in v.values():collect(z)
    elif isinstance(v,list):
        for z in v:collect(z)
collect(architecture)
for path,h in references.items():assert sha(resolve(path))==h;checks+=1
report={'status':'pass','checks':checks,'tables':len(manifest['tables']),'architecture_source_files':len(references),'reconstruction_rows':len(recs),'OD_values':len(od),'MIC_cells':len(mic),'differences':len(diff),'interpretation':'Original reports and retrospective algorithm remain parallel; no adjudication of true MIC.'}
print(json.dumps(report,indent=2))
