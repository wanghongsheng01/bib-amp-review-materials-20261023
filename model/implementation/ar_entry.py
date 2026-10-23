#!/usr/bin/env python3
"""Config-bound AR adapter. Does not select checkpoints or retry."""
import argparse, csv, hashlib, json, pathlib, sys, time, traceback
sys.dont_write_bytecode = True

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x): pathlib.Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def write_slots(path, rows):
    fields=['sample_id','method','terminal','sequence','status','length','seed','terminal_sample_id']
    with pathlib.Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args(); cfg=json.loads(pathlib.Path(a.config).read_text()); out=pathlib.Path(a.output)
    if out.exists(): raise ValueError('fresh output directory required')
    if cfg['implementation_sha256'] != sha(pathlib.Path(__file__).with_name('ar_pipeline.py')): raise ValueError('AR code drift')
    if cfg['entry_sha256'] != sha(__file__): raise ValueError('AR entry drift')
    if cfg['purpose'] not in ('train_only_smoke','formal_pilot'): raise ValueError('purpose')
    smoke=cfg['purpose']=='train_only_smoke'
    if smoke and any(k.startswith('dev') for k in cfg): raise ValueError('smoke forbids dev path')
    out.mkdir(parents=True); dump(out/'config.json',cfg)
    count=1 if smoke else 16
    slots=[{'sample_id':f'terminal_LSTM_{terminal}_{i:03d}','method':'terminal_LSTM','terminal':terminal,'sequence':'','status':'not_executed','length':0,'seed':cfg['sampling_seeds'][terminal],'terminal_sample_id':i} for terminal in ('FREE','AMD') for i in range(count)]
    write_slots(out/'generated.csv',slots)
    status={'state':'running','purpose':cfg['purpose'],'test_read':False,'development_read':False,'config_sha256':sha(a.config)}
    begun=time.perf_counter(); dump(out/'status.json',status)
    try:
        import ar_pipeline as ar
        train=ar.read_split(cfg['train'],role='train',expected_sha256=cfg['train_sha256'],expected_rows=279)
        dev=None
        if not smoke:
            dev=ar.read_split(cfg['dev'],role='development',expected_sha256=cfg['dev_sha256'],expected_rows=41)
            status['development_read']=True
            if {r['sequence'] for r in train['rows']} & {r['sequence'] for r in dev['rows']}: raise ValueError('train/dev overlap')
        kwargs={'train_bundle':train,'shuffle_seed':cfg['shuffle_seed']}
        if not smoke: kwargs.update(dev_bundle=dev,monitor_every=100)
        model,summary=ar.run_train(cfg['training_seconds_cap'],cfg['updates'],cfg['initialization_seed'],out/'training',**kwargs)
        status['training_summary']=summary
        if summary['status']!='COMPLETED_FIXED_STEPS': raise RuntimeError('AR fixed endpoint not completed: '+summary['status'])
        rows,generation=ar.generate(model,cfg['sampling_seeds'],out/'generation',smoke=smoke,cap_seconds=cfg['generation_seconds_cap'])
        status['generation']=generation
        for i,r in enumerate(rows): r['terminal_sample_id']=i%count
        write_slots(out/'generated.csv',rows)
        if not smoke:
            dump(out/'train_nll.json',ar.teacher_forcing_evaluation(model,train))
            dump(out/'dev_nll.json',ar.teacher_forcing_evaluation(model,dev))
        if any(r['status'] not in ('ok','empty_sequence') for r in rows): raise RuntimeError('AR incomplete generation slots')
        status['state']='completed'
    except Exception as e:
        status.update(state='failed',error=repr(e),traceback=traceback.format_exc())
        print(status['traceback'],file=sys.stderr)
    finally:
        status['process_seconds']=time.perf_counter()-begun; dump(out/'status.json',status)
        dump(out/'manifest.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json']})
    print(json.dumps({'method':'terminal_LSTM','state':status['state'],'seconds':status['process_seconds']}),flush=True)
    return 0 if status['state']=='completed' else 2
if __name__=='__main__': raise SystemExit(main())
