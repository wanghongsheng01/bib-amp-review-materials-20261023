#!/usr/bin/env python3
"""Assemble S1-S6 from verified saved outputs; no training, inference or network."""
from pathlib import Path
import csv,json,hashlib,shutil
R=Path(__file__).resolve().parent;B=R.parent
C=B/'compute' if (B/'compute/data').is_dir() else B/'computational/reviewer_candidate'
W=B/'wetlab' if (B/'wetlab/data').is_dir() else B/'wetlab/public_candidate'
T=R/'tables';T.mkdir(exist_ok=True)
bindings=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvread(p):return list(csv.DictReader(p.open()))
def write(name,rows,source,transform='unchanged values'):
    p=T/name
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    bindings.append({'output':str(p.relative_to(R)),'rows':len(rows),'sha256':sha(p),'source':str(source.relative_to(B)),'source_sha256':sha(source),'transformation':transform})
    return rows
def cp(source,name):return write(name,csvread(source),source)
maps={
'S1_reconstruction_train_dev.csv':'Table_S1a_reconstruction_summary.csv',
'S1_reconstruction_per_amino_acid.csv':'Table_S1c_reconstruction_by_amino_acid.csv',
'S2_generation_all_96_distances.csv':'Table_S2a_generation_96_distances.csv',
'S2_generation_24_comparisons.csv':'Table_S2b_generation_24_comparisons.csv',
'independent_generation_96_distances.csv':'Table_S2c_generation_recomputed.csv',
'S3_generation_complementary.csv':'Table_S2d_generation_eligibility_entropy.csv',
'independent_generation_complementary.csv':'Table_S2e_generation_validity_copying.csv',
'S7_native_activity_baselines.csv':'Table_S3a_native_activity.csv',
'S8_censored_activity_baselines.csv':'Table_S3b_censored_activity.csv',
'S8_censored_contrasts.csv':'Table_S3c_tail_endpoint_contrasts.csv',
'S9_activity_variance_decomposition.csv':'Table_S3d_variance_decomposition.csv',
'independent_activity_predictions_aggregated.csv':'Table_S3e_activity_recomputed.csv',
'S4_seven_routes.csv':'Table_S4a_route_classification.csv',
'S4_source_substitution_distances.csv':'Table_S4b_source_substitution_96_distances.csv',
'S4_warmup_guards.csv':'Table_S4c_warmup_guards.csv',
'S4_warmup_paired_metrics.csv':'Table_S4d_warmup_paired_metrics.csv',
'S4_AR_latent_information_gate.csv':'Table_S4e_AR_information.csv',
'S4_direct_z_training.csv':'Table_S4f_direct_latent_training.csv',
'S4_EB_context_complete_metrics.csv':'Table_S4g_conditioning_context.csv',
}
for src,dst in maps.items():cp(C/'results'/src,dst)
src=C/'data/evidence/amp_h5/feasibility/ce_recovery_four_arm_evaluation_v1/reconstruction_row_audit.json'
records=json.loads(src.read_text());assert len(records)==960
per=[]
for x in records:
 t,p=x['truth_sequence'],x['prediction_sequence'];correct=sum(a==p[i] for i,a in enumerate(t) if i<len(p));le=abs(len(t)-len(p))
 assert correct==x['correct_target_positions'] and le==x['length_absolute_error']
 per.append({**x,'target_length':len(t),'prediction_length':len(p),'aligned_accuracy':correct/len(t),'exact_sequence':t==p})
write('Table_S1b_reconstruction_per_peptide.csv',per,src,'JSON to CSV, plus independently calculated lengths, aligned accuracy and exact-match flag')
for src,dst in {
'sample_manifest_30.csv':'Table_S5a_order_delivery_manifest.csv',
'mic_algorithm_calls.csv':'Table_S5b_MIC_reported_algorithm.csv',
'od600_raw.csv':'Table_S5d_OD600_readings.csv',
'coa_27_facts.csv':'Table_S5e_supplier_facts.csv',
'threshold_counts_all.csv':'Table_S5f_all_threshold_counts.csv',
'N_semantics_cells.csv':'Table_S5g_N_semantics.csv',
'group_counts.csv':'Table_S5h_design_group_counts.csv',
'molar_threshold_equivalents.csv':'Table_S6b_molar_mass_equivalents.csv',
}.items():cp(W/'data'/src,dst)
src=W/'data/mic_algorithm_calls.csv'
diffs=[x for x in csvread(src) if x['matches_reported_under_legacy_N_code']=='False']
assert len(diffs)==3
write('Table_S5c_MIC_discrepancies.csv',diffs,src,'Filter to the three reported-versus-algorithm differences; retain every original field')
src=R/'source/AMPGen_publisher_facts.json'
official=json.loads(src.read_text())['checks'][0];f=official['verified_facts']
assert f['initial_active']==31 and f['initial_synthesized']==38 and f['followup_selected_peptides']==9
s6=[
 {'cohort':'AMPGen initial validation','ordered_or_designed':40,'assayed_or_synthesized':38,'numerator':31,'denominator':38,'criterion':'MIC <=75 uM against either initial target','targets':'; '.join(f['initial_endpoints']),'fraction':'31/38 (81.58%)','source':official['url'],'scope':'Published result; no external raw-data reanalysis'},
 {'cohort':'AMPGen selected follow-up','ordered_or_designed':'not applicable','assayed_or_synthesized':9,'numerator':'not reported here','denominator':9,'criterion':'Selected for MIC <5 uM against both initial targets; four-strain follow-up is a separate cohort','targets':'; '.join(f['followup_panel']),'fraction':'Not the denominator of 31/38','source':official['url'],'scope':'Published follow-up panel includes S. aureus ATCC 25923'},
 {'cohort':'Present record: original summary, ordered denominator','ordered_or_designed':30,'assayed_or_synthesized':27,'numerator':6,'denominator':30,'criterion':'S. aureus MIC <200 ug/mL','targets':'S. aureus ATCC 29213','fraction':'6/30 (20.00%)','source':'Table S5','scope':'Original summary'},
 {'cohort':'Present record: original summary, assayed denominator','ordered_or_designed':30,'assayed_or_synthesized':27,'numerator':6,'denominator':27,'criterion':'S. aureus MIC <200 ug/mL','targets':'S. aureus ATCC 29213','fraction':'6/27 (22.22%)','source':'Table S5','scope':'Original summary'},
 {'cohort':'Present record: algorithm, ordered denominator','ordered_or_designed':30,'assayed_or_synthesized':27,'numerator':7,'denominator':30,'criterion':'Retrospective S. aureus endpoint <200 ug/mL','targets':'S. aureus ATCC 29213','fraction':'7/30 (23.33%)','source':'Table S5','scope':'Retrospective calculation; not true-MIC adjudication'},
 {'cohort':'Present record: algorithm, assayed denominator','ordered_or_designed':30,'assayed_or_synthesized':27,'numerator':7,'denominator':27,'criterion':'Retrospective S. aureus endpoint <200 ug/mL','targets':'S. aureus ATCC 29213','fraction':'7/27 (25.93%)','source':'Table S5','scope':'Retrospective calculation; not true-MIC adjudication'},
 {'cohort':'Present record: operational spectrum-label match, ordered','ordered_or_designed':30,'assayed_or_synthesized':27,'numerator':3,'denominator':30,'criterion':'Original design-group operational pattern at <200 ug/mL; N treated as nonqualifying','targets':'Four measured strains','fraction':'3/30 (10.00%)','source':'Table S5h','scope':'Same count under both representations; not proof of specificity'},
 {'cohort':'Present record: operational spectrum-label match, assayed','ordered_or_designed':30,'assayed_or_synthesized':27,'numerator':3,'denominator':27,'criterion':'Original design-group operational pattern at <200 ug/mL; N treated as nonqualifying','targets':'Four measured strains','fraction':'3/27 (11.11%)','source':'Table S5h','scope':'Same count under both representations; not proof of specificity'},
]
write('Table_S6a_AMPGen_comparison.csv',s6,src,'Targeted publisher facts plus Table S5 counts; initial and follow-up AMPGen populations kept separate')
# Additional source bindings for S6 rows derived from this study.
bindings[-1]['additional_sources']=[{'path':str((W/'data'/p).relative_to(B)),'sha256':sha(W/'data'/p)} for p in ['threshold_counts_all.csv','group_counts.csv']]
(R/'table-provenance.json').write_text(json.dumps({'tables':bindings,'computational_source_commit':'f1b748d21386e1d1683a0b6e92cbcf9a208a4d2b','boundary':'Saved-output arithmetic, recovered assay records and factual editing only; no new experiments or MIC adjudication'},indent=2)+'\n')

def esc(s):
 return str(s).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#').replace('≤',r'$\leq$').replace('≥',r'$\geq$')
def fmt(x,n=5):
 try:return f'{float(x):.{n}f}'
 except:return esc(x)
def table(name,caption,headers,rows,spec=None):
 spec=spec or ('l'+'r'*(len(headers)-1))
 spec=spec.replace('p{',r'>{\raggedright\arraybackslash}p{')
 # Longtable prevents silent missing rows and repeats headings on continuation.
 s=r'\begingroup\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}'+'\n'
 s+=r'\begin{longtable}{'+spec+'}\n'+r'\caption*{\textbf{'+name+'.} '+caption+'}'+r'\\'+'\n'+r'\toprule'+'\n'
 head=' & '.join(headers)+r' \\'+ '\n'
 s+=head+r'\midrule\endfirsthead'+'\n'+head+r'\midrule\endhead'+'\n'
 s+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
 s+='\n'+r'\bottomrule\end{longtable}\endgroup'+'\n'
 return s
model={'CE21_DDPM':'CE21','MSE21_DDPM':'MSE21','scalar_DDPM':'Scalar','LSTM_AR':'LSTM','LSTM':'LSTM','terminal_lstm':'LSTM'}
def m(x):return model.get(x,esc(x))
def tr(x):return 'Amide' if x=='AMD' else 'Free'
s1=table('Table S1a','Complete reconstruction summary. Micro accuracy excludes padding; AA macro is the unweighted mean recall of supported amino-acid classes.',
 ['Model','Split','Peptides','Residues','Correct','Micro','AA macro','Length MAE'],
 [[m(x['method']),x['split'],x['rows'],x['target_residues'],x['rendered_aligned_correct'],fmt(x['rendered_nonpad_micro'],4),fmt(x['rendered_nonpad_macro'],4),fmt(x['length_mae'],4)] for x in csvread(T/'Table_S1a_reconstruction_summary.csv')])
aa=csvread(T/'Table_S1c_reconstruction_by_amino_acid.csv')
for split in ['train','development']:
 rs=[]
 for a in 'ACDEFGHIKLMNPQRSTVWY':
  rr=[next(x for x in aa if x['split']==split and x['amino_acid']==a and x['method']==k) for k in ['CE21_DDPM','MSE21_DDPM','scalar_DDPM']]
  rs.append([a,rr[0]['target_residues']]+[x['correct_residues'] for x in rr]+[fmt(x['recall'],4) for x in rr])
 s1+=table('Table S1c ('+split+')','Complete per-amino-acid support, correct counts and recall.', ['AA','Support','CE correct','MSE correct','Scalar correct','CE recall','MSE recall','Scalar recall'],rs)
s1+=r'\noindent Table S1b is supplied as an editable 960-row CSV containing every saved reconstruction (279 training and 41 development peptides for each of three models), including target/predicted sequences, row ID, terminal condition, aligned correct positions, lengths, per-peptide accuracy and exact-match status.'+'\n'
(R/'table_S1.tex').write_text(s1)

all96=csvread(T/'Table_S2a_generation_96_distances.csv');keys=['mmd2_linear_biased_fixed_features','aa_jensen_shannon_bits','aa_l1_distance','dipeptide_jensen_shannon_bits','dipeptide_l1_distance','length_ks_statistic_no_pvalue']
s2=''
for ref in ['train','development']:
 rs=[]
 for mod in dict.fromkeys(x['method'] for x in all96):
  for term in ['FREE','AMD']:
   vals=[next(x['value'] for x in all96 if (x['method'],x['terminal'],x['reference'],x['metric'])==(mod,term,ref,k)) for k in keys]
   rs.append([m(mod),tr(term)]+[fmt(v) for v in vals])
 s2+=table('Table S2a ('+ref+')','All six distribution distances; each model--terminal group contains 16 generation requests. Lower is closer under the statistic.', ['Model','Terminal',r'MMD$^2$','AA JS',r'AA $L_1$','Pair JS',r'Pair $L_1$','Length KS'],rs)
metric_short=dict(zip(keys,[r'MMD$^2$','AA JS',r'AA $L_1$','Pair JS',r'Pair $L_1$','Length KS']))
rs=[]
for x in csvread(T/'Table_S2b_generation_24_comparisons.csv'):
 rs.append([tr(x['terminal']),'Dev' if x['reference']=='development' else 'Train',metric_short[x['metric']]]+[fmt(x[k]) for k in ['CE21','MSE21','scalar','LSTM']])
s2+=table('Table S2b','All 24 aligned method comparisons (two terminal conditions, two reference sets, six statistics).', ['Terminal','Reference','Statistic','CE21','MSE21','Scalar','LSTM'],rs)
rs=[]
for x in csvread(T/'Table_S2d_generation_eligibility_entropy.csv'):
 rs.append([m(x['method']),tr(x['terminal']),x['requests'],x['valid_1_48_count'],x['valid_5_48_count'],x['exact_sequence_copy_any_train_terminal_count'],fmt(x['pooled_aa_entropy_bits_absolute_gap_to_train'],4),fmt(x['pooled_dipeptide_entropy_bits_absolute_gap_to_train'],4)])
s2+=table('Table S2d','Eligibility, exact training copies and absolute entropy gaps in bits.', ['Model','Terminal','Requests','1--48','5--48','Copies','AA gap','Pair gap'],rs)
s2+=r'\noindent Table S2c retains all 96 reported values, independently recomputed values and absolute differences; the maximum absolute difference is $2.45\times10^{-15}$ (rounded upward). Table S2e contains independently recounted validity, uniqueness and exact-copy totals.'+'\n'
(R/'table_S2.tex').write_text(s2)

activity=csvread(T/'Table_S3e_activity_recomputed.csv')
am={'training_weighted_mean':'Mean','composition_ridge':'Ridge','witten_CNN':'CNN','point_mean':'Point mean','point_ridge':'Point ridge','tail_ridge':'Tail','cnn_adaptation':'CNN','qmap_cnn_adaptation':'CNN','qmap_cnn':'CNN','point_only_mean':'Point mean','point_only_ridge':'Point ridge','endpoint_ridge':'Endpoint','censored_gaussian_ridge':'Tail','tail_likelihood_ridge':'Tail'}
s3=''
for exp in dict.fromkeys(x['experiment'] for x in activity):
 rs=[]
 for x in activity:
  if x['experiment']!=exp:continue
  rs.append([r'$\mu$M' if x['unit']=='uM_native' else r'\ug',am.get(x['model'],esc(x['model'])),x['rows']]+[fmt(x[k],4) for k in ['group_peptide_MAE','peptide_MAE','row_MAE','group_peptide_RMSE','peptide_RMSE','row_RMSE']])
 s3+=table('Table S3e ('+('native points' if exp=='native_baseline_v1' else 'censoring screen')+')', 'All six independently recomputed error statistics. G, P and R denote source-group/peptide, peptide-only and record weighting; units are log10 native concentration.', ['Unit','Model','Rows','MAE G','MAE P','MAE R','RMSE G','RMSE P','RMSE R'],rs)
rs=[[r'$\mu$M' if x['unit']=='uM_native' else r'\ug',fmt(x['between_fraction'],6),fmt(x['between_source_group_mean_variance'],6),fmt(x['within_source_group_variance'],6),fmt(x['total_label_variance'],6)] for x in csvread(T/'Table_S3d_variance_decomposition.csv')]
s3+=table('Table S3d','Variance decomposition retained from the source diagnostic.', ['Unit','Between fraction','Between variance','Within variance','Total variance'],rs)
s3+=r'\noindent Tables S3a and S3b retain the original complete model/unit summaries and cohort counts; Table S3c contains all 12 tail-minus-endpoint MAE/RMSE contrasts across the three weighting schemes.'+'\n'
(R/'table_S3.tex').write_text(s3)

routes=csvread(T/'Table_S4a_route_classification.csv')
short=[
 ('Latent substitution','Not applicable','Source diagnostic','Development MMD rises in both terminal groups relative to reused CE.'),
 ('Coordinate permutation','Not applicable','Source diagnostic','Development MMD rises in both groups relative to posterior-source B.'),
 ('Reference-template prior','Not applicable','Source diagnostic','MMD is mixed relative to CE; extra reference information excludes a terminal-only ranking.'),
 ('KL warmup','Not applicable','Generation not evaluated','Four reconstruction/resource guards fail; a separate investment threshold passes.'),
 ('Latent-conditioned AR','Pass','Generation fail','Matched S8 0.690802; evaluation mismatch 0.097533; permuted-training control 0.091958.'),
 ('Direct-latent diffusion','Not applicable','Mechanism screen fail','Sequential value screen not interpreted; checkpoint audit incomplete after safe-loader rejection.'),
 ('Conditioning-context variants','Not applicable','Overall not supported','Some generation guards and selected NLL criteria fail; 384 attempts include three empty outputs.'),
]
assert len(routes)==len(short)==7
s4=table('Table S4a','Route-specific evidence categories. A diagnostic comparison or unperformed generation gate is not labelled a failed generation experiment.', ['Route','Information','Screen classification','Recorded outcome'],[[esc(c) for c in r] for r in short],r'p{.19\linewidth}p{.13\linewidth}p{.20\linewidth}p{.38\linewidth}')
rs=[[esc(x['guard'].replace('_',' ')), 'Pass' if x['pass']=='True' else 'Fail',esc(x['gate_type'])] for x in csvread(T/'Table_S4c_warmup_guards.csv')]
s4+=table('Table S4c','All warmup guards and the separate investment screen.', ['Guard','Outcome','Type'],rs,r'p{.51\linewidth}lp{.32\linewidth}')
eb=csvread(T/'Table_S4g_conditioning_context.csv')
rs=[[esc(x['method']),tr(x['terminal']),x['requests'],x['valid_1_48'],x['valid_5_48'],x['empty'],fmt(x['dev_terminal_sequence_nll'],4)] for x in eb]
s4+=table('Table S4g','Conditioning-context attempt counts and development sequence NLL. Full CV, train/development distances and all recorded guards remain in the editable CSV.', ['Method','Terminal','Attempts','1--48','5--48','Empty','Dev NLL'],rs)
s4+=r'\noindent Tables S4b--S4g retain all 96 source-substitution distances, ten paired warmup statistics, eight information-gate entries, both direct-latent training rows and all 12 model and terminal combinations for the conditioning-context analysis.'+'\n'
(R/'table_S4.tex').write_text(s4)

man=csvread(T/'Table_S5a_order_delivery_manifest.csv');mic=csvread(T/'Table_S5b_MIC_reported_algorithm.csv')
rs=[[esc(x['sample_id']),esc(x['sequence']),x['ordered_group_id'],'Yes' if x['assayed']=='True' else 'No',x['hplc_purity_percent'] or '--'] for x in man]
s5=table('Table S5a','Complete 30-candidate order and 27-sample assay manifest. Group IDs are original order groups; group labels are documentary assignments. An absent assay record does not identify the reason.', ['ID','Sequence','Group','Assayed',r'Purity (\%)'],rs,r'lp{.52\linewidth}rrr')
strain_order=['E. coli ATCC 25922','P. aeruginosa ATCC 27853','S. aureus ATCC 29213','E. faecalis ATCC 29212']
def micv(v):return v if v=='N' else f'{float(v):g}'
rs=[]
for x in man:
 if x['assayed']!='True':continue
 vals=[]
 for strain in strain_order:
  z=next(z for z in mic if z['sample_id']==x['sample_id'] and z['strain']==strain)
  vals.append(micv(z['reported_summary'])+' / '+micv(z['legacy_algorithm_N_code']))
 rs.append([esc(x['sample_id'])]+vals)
s5+=table('Table S5b','All 108 MIC cells in '+r'\ug'+'. Each cell gives the original report followed by the retrospective algorithm. N is a source or legacy algorithm code, not an adjudication of no inhibition.', ['ID','E. coli','P. aeruginosa','S. aureus','E. faecalis'],rs,'lrrrr')
rs=[[esc(x['sample_id']),esc(x['strain'].split(' ATCC')[0]),micv(x['reported_summary']),'/'.join(micv(x['replicate_'+str(i)]) for i in [1,2,3]),micv(x['legacy_algorithm_N_code'])] for x in diffs]
s5+=table('Table S5c','The three located differences. Replicate endpoints and consensus are retrospective calculations, not replacement experimental MIC determinations.', ['ID','Strain','Reported','Three calculated endpoints','Algorithm'],rs,'lllrl')
s5+=r'\noindent Table S5d contains all 3,240 OD600 readings with sample ID, strain, replicate, concentration, negative control and blank values. Tables S5e--S5h contain all 27 supplier fact rows, 320 threshold/source/strain comparisons, 108 N-semantics rows and six source/group count rows.'+'\n'
(R/'table_S5.tex').write_text(s5)

s6tex=table('Table S6a','Cohort and endpoint distinctions in the AMPGen comparison. The 31/38 fraction concerns the initial two-target cohort, not the selected nine-peptide four-strain follow-up.', ['Cohort','Denominator and result','Criterion and strains'],[
 ['AMPGen initial',r'38 synthesized; 31 active',r'MIC $\leq75\ \mu$M against either \textit{E. coli} K88 or \textit{S. aureus} ATCC 29213.'],
 ['AMPGen follow-up','9 selected peptides',r'Selected for MIC $<5\ \mu$M against both initial targets. Follow-up: \textit{E. coli} ATCC 25922, \textit{P. aeruginosa} ATCC 27853, \textit{S. aureus} ATCC 25923 and \textit{E. faecalis} ATCC 29212.'],
 ['Present original summary','6/30; 6/27',r'\textit{S. aureus} ATCC 29213 MIC $<200\ \ug$. Ordered and assayed denominators, respectively.'],
 ['Present retrospective algorithm','7/30; 7/27',r'Same numeric threshold; an algorithmic count, not an adjudicated replacement MIC.'],
 ['Operational spectrum-label match','3/30; 3/27',r'Group-label pattern at $<200\ \ug$ under both representations; N treated as nonqualifying, without proving specificity.'],
 ],r'p{.23\linewidth}p{.22\linewidth}p{.46\linewidth}')
s6tex+=r'\noindent Table S6b provides the 75 $\mu$M-to-mass conversion for each of the 27 present-study peptides using its supplier molecular weight: 80.27625--243.51975 \ug. These are unit conversions for the present peptides, not AMPGen measurements or a harmonized activity comparison.'+'\n'
(R/'table_S6.tex').write_text(s6tex)
print(json.dumps({'status':'pass','csv_tables':len(bindings),'reconstruction_rows':len(per),'MIC_cells':len(mic),'OD_readings':len(csvread(T/'Table_S5d_OD600_readings.csv'))}))
