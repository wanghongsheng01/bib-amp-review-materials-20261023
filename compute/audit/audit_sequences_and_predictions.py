#!/usr/bin/env python3
"""Independent saved-output audit; no source imports/execution, models, network, reserved partitions."""
from pathlib import Path
from collections import Counter,defaultdict
import math,json,csv,hashlib,datetime,re
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results';R=ROOT/'data'
D=json.loads((ROOT/'input_manifest.json').read_text())
H='evidence/amp_h5/';F=H+'feasibility/ce_recovery_four_arm_evaluation_v1/'
used={};checks=[]
def read(p):
 assert not re.search(r'(^|[/_.-])(test|heldout|holdout|test37)([/_.-]|$)',p.lower()),p
 b=(R/p).read_bytes();a=hashlib.sha256(b).hexdigest();e=D['files'][p]
 assert a==e['sha256'] and len(b)==e['bytes'],p
 used[p]={'bytes':len(b),'sha256':a};return b.decode('utf-8-sig')
def js(p):return json.loads(read(p))
def jl(p):return [json.loads(x) for x in read(p).splitlines() if x.strip()]
def ck(label,a,b):
 ok=math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12) if isinstance(a,(float,int)) and isinstance(b,(float,int)) else a==b
 checks.append({'check':label,'pass':ok,'recomputed':a,'reported':b})
def csvout(p,rs):
 with (B/p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
AA='ARNDCEQGHILKMFPSTWYV';DI=[a+b for a in AA for b in AA]
def entropy(c):
 n=sum(c.values());return -sum(v/n*math.log2(v/n) for v in c.values() if v) if n else 0.
def freq(c,keys):
 n=sum(c.values());return [c[k]/n if n else 0 for k in keys]
def stats(seqs):
 ac=Counter(''.join(seqs));dc=Counter(d for s in seqs for d in (s[i:i+2] for i in range(len(s)-1)))
 fm=[0.]*421
 for s in seqs:
  sc=Counter(s);sd=Counter(s[i:i+2] for i in range(len(s)-1));feat=[len(s)/48]+freq(sc,AA)+freq(sd,DI)
  fm=[a+b/len(seqs)/math.sqrt(3) for a,b in zip(fm,feat)]
 return {'aa':freq(ac,AA),'di':freq(dc,DI),'feature':fm,'lengths':list(map(len,seqs)),'aa_entropy':entropy(ac),'di_entropy':entropy(dc)}
def jsdiv(p,q):
 m=[(a+b)/2 for a,b in zip(p,q)]
 return .5*sum(a*math.log2(a/c) if a else 0 for a,c in zip(p,m))+.5*sum(b*math.log2(b/c) if b else 0 for b,c in zip(q,m))
def distance(s,t):
 lens=set(s['lengths']+t['lengths'])
 ks=max(abs(sum(x<=n for x in s['lengths'])/len(s['lengths'])-sum(x<=n for x in t['lengths'])/len(t['lengths'])) for n in lens)
 return {'aa_jensen_shannon_bits':jsdiv(s['aa'],t['aa']),'aa_l1_distance':sum(abs(a-b) for a,b in zip(s['aa'],t['aa'])),'dipeptide_jensen_shannon_bits':jsdiv(s['di'],t['di']),'dipeptide_l1_distance':sum(abs(a-b) for a,b in zip(s['di'],t['di'])),'mmd2_linear_biased_fixed_features':sum((a-b)**2 for a,b in zip(s['feature'],t['feature'])),'length_ks_statistic_no_pvalue':ks}
rec=js(F+'reconstruction_row_audit.json');sumrec=js('manuscripts/review_candidate/evidence/reconstruction_summary.json')['groups']
recs=defaultdict(list)
for x in rec:recs[x['method'],x['split']].append(x)
for group,rs in recs.items():
 g=next(x for x in sumrec if (x['method'],x['split'])==group)
 support=Counter();correct=Counter();micro=0;le=0;lex=0;ex=0
 for x in rs:
  t,p=x['truth_sequence'],x['prediction_sequence'];support.update(t);c=0
  for i,a in enumerate(t):
   if i<len(p) and p[i]==a:correct[a]+=1;c+=1
  ck(str(group)+' row '+x['row_id']+' correct',c,x['correct_target_positions'])
  ck(str(group)+' row '+x['row_id']+' length error',abs(len(t)-len(p)),x['length_absolute_error'])
  micro+=c;le+=abs(len(t)-len(p));lex+=len(t)==len(p);ex+=t==p
 vals={'rows':len(rs),'target_residues':sum(support.values()),'rendered_aligned_correct':micro,'length_absolute_error_sum':le,'length_exact_count':lex,'whole_rendered_exact_count':ex,'rendered_nonpad_macro':sum(correct[a]/support[a] for a in AA if support[a])/sum(support[a]>0 for a in AA)}
 for k,v in vals.items():ck(str(group)+' recomputed '+k,v,g[k])
 for a in AA:ck(str(group)+' support '+a,support[a],g['target_support_by_aa'][a]);ck(str(group)+' correct '+a,correct[a],g['rendered_correct_by_aa'][a])
refs=defaultdict(list)
for x in rec:
 if x['method']=='CE21_DDPM':refs[x['terminal'],x['split']].append(x['truth_sequence'])
ck('train/dev reference counts',sorted(len(x) for x in refs.values()),[18,23,126,153])
trainset=set(refs['FREE','train']+refs['AMD','train']);gs=js(F+'generation_sample_audit.json');sg=js(F+'generation_metrics.json')['groups'];out=[];complement=[]
for g in sg:
 rs=[x for x in gs if (x['method'],x['terminal'])==(g['method'],g['terminal'])];seqs=[x['sequence'] for x in rs]
 valid=[s for s in seqs if 1<=len(s)<=48 and set(s)<=set(AA)];s=stats(valid)
 vals={'requests':len(rs),'valid_1_48_count':len(valid),'valid_5_48_count':sum(5<=len(x)<=48 for x in valid),'exact_sequence_copy_any_train_terminal_count':sum(x in trainset for x in seqs)}
 for k,v in vals.items():ck(g['method']+' '+g['terminal']+' actual sequences '+k,v,g[k])
 ck(g['method']+' '+g['terminal']+' unique valid',len(set(valid)),g['valid_1_48']['unique_count'])
 for split in ['train','development']:
  t=stats(refs[g['terminal'],split]);ds=distance(s,t)
  for k,v in ds.items():
   rep=g['valid_1_48_vs_'+split][k];ck('sequence distance '+g['method']+' '+g['terminal']+' '+split+' '+k,v,rep)
   out.append({'method':g['method'],'terminal':g['terminal'],'reference':split,'metric':k,'recomputed':v,'reported':rep,'absolute_error':abs(v-rep)})
  if split=='train':
   for k,sk in [('pooled_aa_entropy_bits_absolute_gap_to_train','aa_entropy'),('pooled_dipeptide_entropy_bits_absolute_gap_to_train','di_entropy')]:ck(g['method']+' '+g['terminal']+' '+k,abs(s[sk]-t[sk]),g[k])
 complement.append({'method':g['method'],'terminal':g['terminal'],**vals,'unique_valid_count':len(set(valid))})
csvout('independent_generation_96_distances.csv',out);csvout('independent_generation_complementary.csv',complement)
# Initial states, actual update schedules and parameter scope.
params=[];traces={}
for arm in ['categorical_ce','raw_onehot_mse']:
 p=H+'runs/formal_v2/'+arm+'/'
 init=js(p+'vae_initialization.json');ddpm=js(p+'ddpm_training_summary.json');tr=jl(p+'vae_train.jsonl');traces[arm]=(init,tr)
 ck(arm+' VAE update steps',[x['step'] for x in tr],list(range(1,1001)))
 ck(arm+' DDPM updates',ddpm['updates'],1000)
 params.append({'arm':arm,'vae_parameters':init['parameters'],'vae_decoder_head_parameters':init['decoder_head_parameters'],'ddpm_parameters':ddpm['parameters'],'total_vae_ddpm_parameters':init['parameters']+ddpm['parameters']})
a,b=[traces[x] for x in ['categorical_ce','raw_onehot_mse']]
ck('matched VAE initial tensor hash',a[0]['tensor_state_sha256'],b[0]['tensor_state_sha256'])
for key in ['record_indices','forward_rng_sha256','lr']:
 ck('matched all1000 '+key,[x[key] for x in a[1]],[x[key] for x in b[1]])
csvout('model_parameter_scope.csv',params)
mon=js(H+'runs/formal_v1/terminal_lstm/training/MONITORS.json');csvout('LSTM_development_NLL_by_update.csv',[{'step':x['step'],'mean_nll':x['development']['mean_nll'],'used_for_selection':x['used_for_selection']} for x in mon])
ck('AR step200 NLL rounded',round(next(x for x in mon if x['step']==200)['development']['mean_nll'],4),2.6153)
ck('AR endpoint NLL rounded',round(mon[-1]['development']['mean_nll'],4),3.1698)
# Independently recompute all six activity error statistics from saved held-source-group development predictions.
activity=[]
for experiment in ['native_baseline_v1','censored_likelihood_v1']:
 preds=jl(H+experiment+'/run/PREDICTIONS.jsonl');summary=js(H+experiment+'/run/RESULT.json')
 grouped=defaultdict(list)
 for r in preds:grouped[r['native_unit_group'],r['model']].append(r)
 for (unit,model),rs in grouped.items():
  peptides=defaultdict(list);groups=defaultdict(lambda:defaultdict(list))
  for r in rs:
   e=r['prediction_log10']-r['target_log10'];peptides[r['peptide_id']].append(e);groups[r['source_family_group']][r['peptide_id']].append(e)
  def mean(x):return sum(x)/len(x)
  vals={}
  for loss in ['MAE','RMSE']:
   fn=(lambda e:abs(e)) if loss=='MAE' else (lambda e:e*e)
   row=mean([fn(r['prediction_log10']-r['target_log10']) for r in rs]);pep=mean([mean([fn(e) for e in es]) for es in peptides.values()]);grp=mean([mean([mean([fn(e) for e in es]) for es in peps.values()]) for peps in groups.values()])
   if loss=='RMSE':row,pep,grp=map(math.sqrt,[row,pep,grp])
   prefix='group_peptide_macro_' if experiment=='native_baseline_v1' else 'group_peptide_';pp='peptide_macro_' if experiment=='native_baseline_v1' else 'peptide_'
   vals.update({prefix+loss:grp,pp+loss:pep,'row_'+loss:row})
  target=summary['units'][unit][model]
  for k,v in vals.items():ck(experiment+' '+unit+' '+model+' '+k,v,target[k])
  activity.append({'experiment':experiment,'unit':unit,'model':model,'rows':len(rs),'peptides':len(peptides),'groups':len(groups),'group_peptide_MAE':next(v for k,v in vals.items() if k.startswith('group_') and k.endswith('MAE')),'peptide_MAE':next(v for k,v in vals.items() if k.startswith('peptide_') and k.endswith('MAE')),'row_MAE':vals['row_MAE'],'group_peptide_RMSE':next(v for k,v in vals.items() if k.startswith('group_') and k.endswith('RMSE')),'peptide_RMSE':next(v for k,v in vals.items() if k.startswith('peptide_') and k.endswith('RMSE')),'row_RMSE':vals['row_RMSE']})
csvout('independent_activity_predictions_aggregated.csv',activity)
# Large equality checks retain concise hash rather than raw arrays in validation receipt.
for c in checks:
 for k in ['recomputed','reported']:
  if isinstance(c[k],list) and len(c[k])>50:c[k]={'items':len(c[k]),'sha256':hashlib.sha256(json.dumps(c[k],sort_keys=True).encode()).hexdigest()}
receipt={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':D['commit'],'scope':'Independent arithmetic from960 saved train/dev reconstruction sequences;128 saved generation sequences;255 native and588 censored development predictions; JSON metadata and1000step traces. No source code execution, no training, inference, weights, network or reserved test access.','checks':checks,'pass_count':sum(x['pass'] for x in checks),'fail_count':sum(not x['pass'] for x in checks),'source_files':used}
(B/'sequence_prediction_verification.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({'checks':len(checks),'pass':receipt['pass_count'],'fail':[c for c in checks if not c['pass']],'files':len(used),'max_generation_abs_error':max(x['absolute_error'] for x in out)}))
