#!/usr/bin/env python3
"""Independently aggregate explicitly selected saved train/development outputs.
No model imports, execution, checkpoint unpickling, network, test data or source-code execution.
"""
from pathlib import Path
from collections import defaultdict
import csv, json, math, hashlib, datetime, re
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results'
REPO=ROOT/'data'
DISCOVERY=json.loads((ROOT/'input_manifest.json').read_text())
E='manuscripts/review_candidate/evidence/'
used={}; checks=[]
def read(rel):
    assert not re.search(r'(^|[/_.-])(test|heldout|holdout|test37)([/_.-]|$)',rel.lower()),rel
    p=REPO/rel;b=p.read_bytes();expected=DISCOVERY['files'][rel]
    actual=hashlib.sha256(b).hexdigest()
    assert actual==expected['sha256'] and len(b)==expected['bytes'],rel
    used[rel]={'bytes':len(b),'sha256':actual}
    return b.decode('utf-8-sig')
def js(rel):return json.loads(read(rel))
def check(label,condition,details=None):
    checks.append({'check':label,'pass':bool(condition),'details':details})
def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
def csvout(name,rows):
    if not rows:return
    with (BASE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

rec=js(E+'reconstruction_summary.json')['groups']
reconstruction=[];peraa=[]
for g in rec:
    support=g['target_support_by_aa'];correct=g['rendered_correct_by_aa']
    denom=sum(support.values());numer=sum(correct.values())
    check(f"{g['method']} {g['split']} per-aa micro",denom==g['target_residues'] and numer==g['rendered_aligned_correct'] and close(numer/denom,g['rendered_nonpad_micro']))
    macro=sum(correct[aa]/n for aa,n in support.items() if n)/sum(bool(n) for n in support.values())
    check(f"{g['method']} {g['split']} macro from per-aa aggregates",close(macro,g['rendered_nonpad_macro']))
    reconstruction.append({k:g[k] for k in ['method','split','rows','target_residues','rendered_aligned_correct','rendered_nonpad_micro','rendered_nonpad_macro','length_absolute_error_sum','length_mae','length_exact_count','whole_rendered_exact_count']})
    for aa,n in support.items():peraa.append({'method':g['method'],'split':g['split'],'amino_acid':aa,'target_residues':n,'correct_residues':correct[aa],'recall':correct[aa]/n if n else ''})
for name,method in [('CE','CE21_DDPM'),('MSE','MSE21_DDPM'),('scalar','scalar_DDPM')]:
    rows=js(E+f'dev_reconstruction_{name}.json');g=next(x for x in rec if x['split']=='development' and x['method']==method)
    vals={'rows':len(rows),'target_residues':sum(r['true_length'] for r in rows),'rendered_aligned_correct':sum(r['correct_residues'] for r in rows),'length_absolute_error_sum':sum(abs(r['true_length']-r['prediction_length']) for r in rows),'whole_rendered_exact_count':sum(r['exact'] for r in rows),'length_exact_count':sum(r['true_length']==r['prediction_length'] for r in rows)}
    for k,v in vals.items():check(f'{method} dev rows aggregate {k}',v==g[k],{'recomputed':v,'source_summary':g[k]})
    check(f'{method} dev unique row keys',len({r['row_key'] for r in rows})==41)
    check(f'{method} dev terminal counts',sum(r['terminal']=='FREE' for r in rows)==18 and sum(r['terminal']=='AMD' for r in rows)==23)
csvout('S1_reconstruction_train_dev.csv',reconstruction);csvout('S1_reconstruction_per_amino_acid.csv',peraa)
dev={r['method']:r for r in rec if r['split']=='development'}
check('v2 Table1 rounded micro values',[round(100*dev[x]['rendered_nonpad_micro'],2) for x in ['CE21_DDPM','MSE21_DDPM','scalar_DDPM']]==[70.90,12.55,15.83])
check('v2 difference 58.35 percentage points',round(100*(dev['CE21_DDPM']['rendered_nonpad_micro']-dev['MSE21_DDPM']['rendered_nonpad_micro']),2)==58.35)
check('CE eight zero-recall classes',[aa for aa,n in dev['CE21_DDPM']['target_support_by_aa'].items() if n and dev['CE21_DDPM']['rendered_correct_by_aa'][aa]==0]==list('CDEHMNQY'))
gen=js(E+'generation_summary.json');csv_rows=list(csv.DictReader(read(E+'generation_all_distances.csv').splitlines()))
lookup={(r['method'],r['terminal'],r['reference'],r['metric']):float(r['value']) for r in csv_rows}
check('complete 96-distance unique records',len(lookup)==len(csv_rows)==96)
metrics=list(gen[0]['valid_1_48_vs_train'])
for g in gen:
 for ref in ['train','development']:
  for m,v in g['valid_1_48_vs_'+ref].items():check('CSV/JSON '+str((g['method'],g['terminal'],ref,m)),close(v,lookup[g['method'],g['terminal'],ref,m]))
comparisons=[]
for terminal in ['FREE','AMD']:
 for ref in ['train','development']:
  for metric in metrics:
   ce,mse,scalar,ar=[lookup[m,terminal,ref,metric] for m in ['CE21_DDPM','MSE21_DDPM','scalar_DDPM','terminal_LSTM']]
   comparisons.append({'terminal':terminal,'reference':ref,'metric':metric,'CE21':ce,'MSE21':mse,'scalar':scalar,'LSTM':ar,'CE21_lt_MSE21':ce<mse,'CE21_lt_scalar':ce<scalar,'LSTM_lt_CE21':ar<ce})
check('CE improves all 24 distances over MSE',all(r['CE21_lt_MSE21'] for r in comparisons))
check('CE improves all 24 distances over scalar',all(r['CE21_lt_scalar'] for r in comparisons))
check('LSTM closer than CE on all 24 distances',all(r['LSTM_lt_CE21'] for r in comparisons))
csvout('S2_generation_all_96_distances.csv',csv_rows);csvout('S2_generation_24_comparisons.csv',comparisons)
complement=[]
for g in gen:complement.append({k:g[k] for k in ['method','terminal','requests','valid_1_48_count','valid_5_48_count','exact_sequence_copy_any_train_terminal_count','pooled_aa_entropy_bits_absolute_gap_to_train','pooled_dipeptide_entropy_bits_absolute_gap_to_train']})
csvout('S3_generation_complementary.csv',complement)
check('128 main requests, 16 per group',sum(g['requests'] for g in gen)==128 and all(g['requests']==16 for g in gen))
check('all 128 canonical length 1-48',all(g['valid_1_48_count']==16 for g in gen))
check('no exact training sequence copies',all(g['exact_sequence_copy_any_train_terminal_count']==0 for g in gen))
for terminal in ['FREE','AMD']:
 a={g['method']:g for g in gen if g['terminal']==terminal}
 for key in ['pooled_aa_entropy_bits_absolute_gap_to_train','pooled_dipeptide_entropy_bits_absolute_gap_to_train']:
  check(f'scalar lower entropy gap than CE {terminal} {key}',a['scalar_DDPM'][key]<a['CE21_DDPM'][key])

native=js(E+'native_baseline_v1_summary.json');nr=[]
for unit,models in native['units'].items():
 for model,vals in models.items():nr.append({'unit':unit,'model':model,**vals})
check('native folds 17',len(native['completed_folds'])==17)
check('native CNN updates 1700',native['total_CNN_updates']==1700)
for unit,models in native['units'].items():
 check(f'CNN primary MAE worse than mean {unit}',models['witten_CNN']['group_peptide_macro_MAE']>models['training_weighted_mean']['group_peptide_macro_MAE'])
check('mass ridge direction flips with weighting',native['units']['ug_ml_native']['composition_ridge']['group_peptide_macro_MAE']<native['units']['ug_ml_native']['training_weighted_mean']['group_peptide_macro_MAE'] and native['units']['ug_ml_native']['composition_ridge']['peptide_macro_MAE']>native['units']['ug_ml_native']['training_weighted_mean']['peptide_macro_MAE'] and native['units']['ug_ml_native']['composition_ridge']['row_MAE']>native['units']['ug_ml_native']['training_weighted_mean']['row_MAE'])
csvout('S7_native_activity_baselines.csv',nr)
censored=js(E+'censored_likelihood_v1_summary.json');cr=[];cc=[]
for unit,models in censored['units'].items():
 for model,vals in models.items():cr.append({'unit':unit,'model':model,**vals})
 for metric in ['group_peptide_MAE','peptide_MAE','row_MAE','group_peptide_RMSE','peptide_RMSE','row_RMSE']:
  d=models['tail_ridge'][metric]-models['endpoint_ridge'][metric]
  cc.append({'unit':unit,'metric':metric,'tail_minus_endpoint':d});check(f'tail worse than endpoint {unit} {metric}',d>0)
check('censored 34 point-validation folds',len(censored['completed_folds'])==34)
check('censored 588 point predictions',sum(v['rows'] for m in censored['units'].values() for v in m.values())==588)
check('censored main difference mass',round(censored['units']['ug_ml_native']['tail_ridge']['group_peptide_MAE']-censored['units']['ug_ml_native']['endpoint_ridge']['group_peptide_MAE'],6)==0.023094)
check('censored main difference molar',round(censored['units']['uM_native']['tail_ridge']['group_peptide_MAE']-censored['units']['uM_native']['endpoint_ridge']['group_peptide_MAE'],6)==0.137381)
csvout('S8_censored_activity_baselines.csv',cr);csvout('S8_censored_contrasts.csv',cc)
rootcause=js('evidence/amp_h5/root_cause_diagnostic_v1/SUMMARY.json');rc=[]
for unit,r in rootcause.items():
 v=r['variance_decomposition'];rc.append({'unit':unit,**v})
 check('variance decomposition identity '+unit,close(v['between_source_group_mean_variance']+v['within_source_group_variance'],v['total_label_variance']) and close(v['between_fraction'],v['between_source_group_mean_variance']/v['total_label_variance']))
csvout('S9_activity_variance_decomposition.csv',rc)
eb=list(csv.DictReader(read('manuscripts/EB_addendum/AMP_H5_EB_context_addendum_metrics.csv').splitlines()));csvout('S4_EB_context_complete_metrics.csv',eb)
check('EB 384 requests',sum(int(r['requests']) for r in eb)==384)
for method,target in [('partial',49.16431),('pooled',48.58710),('separate',48.91772),('ar_cv',48.04302),('ar_historical',57.96692)]:
 rs=[r for r in eb if r['method']==method]
 check('EB sequence NLL '+method,len(rs)==2 and all(round(float(r['dev_equal_terminal_sequence_nll']),5)==target for r in rs))

arl=js(E+'ar_latent_v3_summary.json');dz=js(E+'direct_z_impl_v1_summary.json')
check('ARlatent information gate PASS generation FAIL',arl['information_gate']['status']=='PASS' and arl['generation_gate']=='FAIL')
for key,target in [('A_S8',.690802),('B_S8',.091958),('A_mismatch_S8',.097533)]:check('ARlatent '+key,round(arl['information_gate'][key],6)==target)
check('direct-z completed1000 each',all(x['completed_updates']==1000 for x in dz['training'].values()))
check('direct-z mechanism FAIL',dz['screen_status']=='FAIL_MECHANISM_SCREEN')
csvout('S4_AR_latent_information_gate.csv',[{'metric':k,'value':v} for k,v in arl['information_gate'].items() if isinstance(v,(int,float,str))])
csvout('S4_direct_z_training.csv',[{'arm':arm,**vals} for arm,vals in dz['training'].items()])
configs={n:js(E+n) for n in ['runs_formal_v2_raw_onehot_mse_config.json','runs_formal_v1_terminal_lstm_config.json']}
(BASE/'frozen_config_excerpt.json').write_text(json.dumps(configs,indent=2))

(BASE/'verification.json').write_text(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Independent arithmetic aggregation of saved train/development summaries and per-row outcome counts; not rerunning training or inference; no test access','source_repository':DISCOVERY['repository'],'source_commit':DISCOVERY['commit'],'checks':checks,'pass_count':sum(x['pass'] for x in checks),'fail_count':sum(not x['pass'] for x in checks),'source_files':used},indent=2))
print(json.dumps({'checks':len(checks),'pass':sum(x['pass'] for x in checks),'failures':[x for x in checks if not x['pass']],'files':len(used)}))
