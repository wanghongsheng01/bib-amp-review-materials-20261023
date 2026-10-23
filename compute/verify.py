#!/usr/bin/env python3
"""Single entry point: Python3 standard library; offline; saved outputs only."""
from pathlib import Path
import hashlib,json,runpy,sys
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'input_manifest.json').read_text())
for name,entry in manifest['files'].items():
 p=ROOT/'data'/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
  raise SystemExit('Input integrity failed: '+name)
(ROOT/'results').mkdir(exist_ok=True)
for script in ['audit_saved_evidence.py','audit_sequences_and_predictions.py','extract_route_evidence.py']:
 runpy.run_path(str(ROOT/'audit'/script),run_name='__main__')
receipts=[json.loads((ROOT/'results'/f).read_text()) for f in ['verification.json','sequence_prediction_verification.json']]
qa={'input_files':len(manifest['files']),'checks':sum(r['pass_count']+r['fail_count'] for r in receipts),'passed':sum(r['pass_count'] for r in receipts),'failed':sum(r['fail_count'] for r in receipts),'scope':'Saved-output arithmetic and reported metadata only; no model training, inference, checkpoints, network or reserved evaluation partition.'}
(ROOT/'results'/'QA.json').write_text(json.dumps(qa,indent=2)+'\n')
print(json.dumps(qa))
sys.exit(1 if qa['failed'] else 0)
