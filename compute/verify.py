#!/usr/bin/env python3
"""Verify saved outputs offline in a temporary workspace; keep released files fixed."""
from pathlib import Path
import hashlib,json,subprocess,sys,tempfile,shutil
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'input_manifest.json').read_text())
for name,entry in manifest['files'].items():
 p=ROOT/'data'/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
  raise SystemExit('Input integrity failed: '+name)
with tempfile.TemporaryDirectory(prefix='amp-offline-check-') as temporary:
 work=Path(temporary)
 for directory in ['audit','data']:
  shutil.copytree(ROOT/directory,work/directory)
 shutil.copy2(ROOT/'input_manifest.json',work/'input_manifest.json')
 (work/'results').mkdir()
 for script in ['audit_saved_evidence.py','audit_sequences_and_predictions.py','extract_route_evidence.py']:
  subprocess.run([sys.executable,str(work/'audit'/script)],check=True)
 receipts=[json.loads((work/'results'/f).read_text()) for f in ['verification.json','sequence_prediction_verification.json']]
 qa={'input_files':len(manifest['files']),'checks':sum(r['pass_count']+r['fail_count'] for r in receipts),'passed':sum(r['pass_count'] for r in receipts),'failed':sum(r['fail_count'] for r in receipts),'scope':'Saved-output arithmetic and reported metadata only; no model training, inference, checkpoints, network or reserved evaluation partition.'}
print(json.dumps(qa))
sys.exit(1 if qa['failed'] else 0)
