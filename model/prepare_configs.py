#!/usr/bin/env python3
"""Materialize local paths in frozen settings; does not import or run model code."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]
OUT=R/'replay_configs';OUT.mkdir(exist_ok=True)
for arm in ('categorical_ce','raw_onehot_mse','scalar_legacy','terminal_lstm'):
 c=json.loads((R/'model/configs'/f'{arm}.json').read_text())
 for k in ('source_root','train','dev','train_cohort','dev_cohort'):
  if k not in c:continue
  p=(R/c[k]).resolve();p.relative_to(R)
  if k!='source_root':assert hashlib.sha256(p.read_bytes()).hexdigest()==c[k+'_sha256']
  c[k]=str(p)
 implementation=R/'model/implementation'/('ar_pipeline.py' if arm=='terminal_lstm' else 'vae_pipeline_formal_v2.py')
 assert hashlib.sha256(implementation.read_bytes()).hexdigest()==c['implementation_sha256']
 if arm=='terminal_lstm':assert hashlib.sha256((R/'model/implementation/ar_entry.py').read_bytes()).hexdigest()==c['entry_sha256']
 (OUT/f'{arm}.json').write_text(json.dumps(c,indent=2)+'\n')
print('Four frozen configurations resolved locally; no model code executed.')
