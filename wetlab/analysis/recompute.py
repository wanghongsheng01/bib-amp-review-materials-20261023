#!/usr/bin/env python3
"""Recompute algorithmic endpoints and tabulations without changing supplied data."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,math,hashlib,sys
R=Path(__file__).resolve().parents[1];D=R/'data'
def rows(n):return list(csv.DictReader((D/n).open()))
def canon(x):return 'N' if x=='N' else float(x)
def equal(a,b):return a==b if isinstance(a,str) or isinstance(b,str) else abs(a-b)<1e-9
names={'ecoli':'E. coli ATCC 25922','paeruginosa':'P. aeruginosa ATCC 27853','saureus':'S. aureus ATCC 29213','efaecalis':'E. faecalis ATCC 29212'}
raw=rows('od600_raw.csv');assert len(raw)==3240
replicates=defaultdict(dict)
for x in raw:
 key=(x['sample_id'],x['strain'],int(x['replicate']));c=float(x['concentration_ug_ml']);v=float(x['od600']);assert math.isfinite(v);assert c not in replicates[key];replicates[key][c]=v
assert len(replicates)==324
replicate_calls={}
for key,v in replicates.items():
 assert set(v)=={1000/2**j for j in range(10)}
 vals=[c for c,od in v.items() if od<0.10];replicate_calls[key]=min(vals) if vals else 'N'
alg={};statuses={}
for s,k,_ in replicates:
 if (s,k) in alg:continue
 rs=[replicate_calls[s,k,i] for i in [1,2,3]];counts=Counter(rs);m,n=counts.most_common(1)[0];alg[s,k]=m if n>=2 else 'N';statuses[s,k]='no_consensus' if n<2 else ('no_endpoint_majority' if m=='N' else 'numeric_consensus')
assert len(alg)==108
for x in rows('mic_algorithm_calls.csv'):
 k=next(k for k,n in names.items() if n==x['strain']);key=(x['sample_id'],k);assert equal(alg[key],canon(x['legacy_algorithm_N_code']));assert statuses[key]==x['algorithm_status']
reported={}
for x in rows('mic_summary_reported.csv'):
 for k in names:reported[x['sample_id'],k]=canon(x[k+'_ug_ml'])
assert len(reported)==108
ids=sorted({s for s,k in alg})
def active(v,t,op):return v!='N' and (v<t if op=='<' else v<=t)
for x in rows('threshold_counts_all.csv'):
 vals=reported if x['source']=='reported_summary' else alg;t=float(x['threshold_ug_ml']);op=x['comparison'];ks=list(names) if x['strain']=='any_of_four' else [next(k for k,n in names.items() if n==x['strain'])];hit=[s for s in ids if any(active(vals[s,k],t,op) for k in ks)];assert len(hit)==int(x['numeric_hits']);assert ';'.join(hit)==x['sample_ids']
manifest=rows('sample_manifest_30.csv');assert len(manifest)==30;assert {x['sample_id'] for x in manifest if x['assayed']=='True'}==set(ids)
coas=rows('coa_27_facts.csv');om={x['sample_id']:x['sequence'] for x in manifest};assert len(coas)==27
for x in coas:assert om[x['sample_id']]==x['sequence']
assert sum(float(x['hplc_purity'].rstrip('%'))>=95 for x in coas)==26
assert sorted(x['sample_id'] for x in manifest if x['assayed']=='False')==['pp1-6','pp3-10','pp3-8']
differences=[{'sample_id':s,'strain':names[k],'reported':reported[s,k],'algorithm':alg[s,k]} for s,k in alg if not equal(alg[s,k],reported[s,k])]
report={'status':'pass','od_values':len(raw),'replicate_series':len(replicates),'endpoint_cells':len(alg),'matched_cells':len(alg)-len(differences),'differences':differences,'algorithm_status_counts':dict(Counter(statuses.values())),'threshold_rows_verified':len(rows('threshold_counts_all.csv')),'ordered':30,'assayed':27,'COA_sequence_matches':27,'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.iterdir()) if p.is_file()},'interpretation':'Algorithmic consistency only; not an adjudication of true MIC, prospective endpoint choice or model attribution.'}
print(json.dumps(report,indent=2))
