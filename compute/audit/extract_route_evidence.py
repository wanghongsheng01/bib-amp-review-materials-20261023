#!/usr/bin/env python3
"""Extract exact already-completed route evidence; no model or source execution."""
import json,csv,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results';R=ROOT/'data';D=json.loads((ROOT/'input_manifest.json').read_text());H='evidence/amp_h5/';sources={}
def js(p):
 b=(R/p).read_bytes();a=hashlib.sha256(b).hexdigest();assert a==D['files'][p]['sha256'];sources[p]={'sha256':a,'bytes':len(b)};return json.loads(b)
def out(p,rows):
 with (B/p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
p=H+'feasibility/ce_source_replacement_evaluation_v2/generation_metrics.json';sg=js(p)['groups'];rows=[]
for g in sg:
 for ref in ['train','development']:
  for metric,value in g['valid_1_48_vs_'+ref].items():rows.append({'method':g['method'],'terminal':g['terminal'],'reference':ref,'metric':metric,'value':value,'source':p})
out('S4_source_substitution_distances.csv',rows)
interp=H+'feasibility/ce_source_replacement_author_readout_v2/GATE_AND_SCOPE_INTERPRETATION.json';j=js(interp)
warm=H+'night_followup_v1/warmup_paired_guard_run_v1/PAIRED_RESOURCE_GUARD.json';w=js(warm);inv=H+'night_followup_v1/warmup_scoring_run_v1/INVESTMENT_SCREEN.json';i=js(inv)
out('S4_warmup_paired_metrics.csv',[{'arm':arm,'metric':key,**val,'source':warm} for arm,vals in w['exact_values'].items() for key,val in vals.items()])
out('S4_warmup_guards.csv',[{'guard':k,'pass':v,'gate_type':'paired reconstruction/resource guard','source':warm} for k,v in w['guards'].items()]+[{'guard':'M_minus_S8_gt_0.05','pass':i['crosses_investment_threshold'],'gate_type':'separate investment screen; not generation success','source':inv}])
a=H+'ar_latent_v3/runs/natural_v1/SUMMARY.json';ar=js(a)
d=H+'direct_z_impl_v1/DELIVERY_DIRECT_Z_v3.json';dz=js(d)
metrics=dz['runner_reported_metrics'];(B/'S4_direct_z_exact_metrics.json').write_text(json.dumps(metrics,indent=2))
e=H+'eb_context_v1/natural_run/RESULT.json';eb=js(e);eg=H+'eb_context_v1/natural_run/GENERATIONS.json';gen=js(eg)
print('EB ledger',type(gen).__name__,len(gen), 'directzmetrics',type(metrics).__name__)
routes=[]
for name,method,note in [('latent substitution','EMPIRICAL_POSTERIOR_B','Development MMD increases versus reused CE in both terminal groups'),('coordinate permutation','COORDINATE_PERMUTATION_C','Development MMD increases versus empirical-posterior B in both terminal groups'),('reference-template prior','REFERENCE_G_PRIOR_D','Mixed MMD versus reused CE; receives additional reference-g information and excluded from c-only ranking')]:
 routes.append({'route':name,'information_gate':'NOT_APPLICABLE','generation_gate':'NOT_APPLICABLE_SOURCE_DIAGNOSTIC','other_gate':'No new winner gate','observed_result':note,'exact_status':j['legacy_gate_artifact_interpretation'],'source':interp,'metric_source':p})
routes += [
 {'route':'KL warmup','information_gate':'NOT_APPLICABLE','generation_gate':'NOT_EVALUATED','other_gate':'4 reconstruction/resource guards FAIL; separate investment threshold PASS','observed_result':'M 0.7227961128580324 to 0.6578523604648727; S8 0.6611157984837004 to 0.5890993500883396','exact_status':str(w['resource_success']),'source':warm,'metric_source':inv},
 {'route':'latent-conditioned autoregressive','information_gate':ar['information_gate']['status'],'generation_gate':ar['generation_gate'],'other_gate':'Information and generation gates are distinct','observed_result':'A_S8 0.6908023670646612; B_S8 0.09195780008237533; A_mismatch_S8 0.09753266616727502','exact_status':'information PASS; generation FAIL','source':a,'metric_source':a},
 {'route':'direct-latent diffusion','information_gate':'NOT_APPLICABLE','generation_gate':'MECHANISM_SCREEN_FAIL; VALUE_SCREEN_NOT_INTERPRETED','other_gate':'checkpoint audit INCOMPLETE after safe-loader rejection; saved metrics passed historical independent audit','observed_result':'Z worse than R on dev MMD and dipeptide JS for FREE and AMD','exact_status':dz['runner_scientific_outcome'],'source':d,'metric_source':d},
 {'route':'conditioning-context variants','information_gate':'NOT_APPLICABLE','generation_gate':'Some generation guards FAIL; overall multistage screen NOT_SUPPORTED','other_gate':'train grouped CV and dev NLL also fail selected criteria','observed_result':'384 attempts in saved ledger; partial dev balanced sequence NLL 49.164308616495234','exact_status':eb['status'],'source':e,'metric_source':eg}]
out('S4_seven_routes.csv',routes)
g=H+'feasibility/ce_recovery_four_arm_evaluation_v1/exploratory_gates.json';gates=js(g);(B/'primary_exploratory_gates.json').write_text(json.dumps(gates,indent=2))
(B/'route_source_provenance.json').write_text(json.dumps({'source_commit':D['commit'],'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':sources,'EB_saved_ledger_length':len(gen),'note':'Descriptive extraction; source scripts not executed; generation-gate nonapplicability retained.'},indent=2))
