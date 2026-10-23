#!/usr/bin/env python3
"""Offline integrity and saved-output calculations; standard-library only."""
from pathlib import Path
import hashlib,json,subprocess,sys,ast
R=Path(__file__).resolve().parent
m=json.loads((R/'MANIFEST.json').read_text())
for e in m['files']:
 p=R/e['path']
 if p.is_symlink() or not p.is_file() or p.stat().st_size!=e['bytes']:raise SystemExit('File integrity failed: '+e['path'])
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 if h.hexdigest()!=e['sha256']:raise SystemExit('Hash mismatch: '+e['path'])
for p in (R/'model').rglob('*.py'):ast.parse(p.read_text(),filename=str(p.relative_to(R)))
calc=subprocess.run([sys.executable,str(R/'compute/verify.py')],cwd=R,check=True,text=True,capture_output=True)
comp=json.loads(calc.stdout.splitlines()[-1])
lab=subprocess.run([sys.executable,str(R/'wetlab/analysis/recompute.py')],cwd=R,check=True,text=True,capture_output=True)
wet=json.loads(lab.stdout)
supp=subprocess.run([sys.executable,str(R/'supplement/verify_supplement.py')],cwd=R,check=True,text=True,capture_output=True)
sup=json.loads(supp.stdout)
result={'supplementary_checks':sup['checks'],'supplementary_status':sup['status'],'integrity_files':len(m['files']),'computational_checks':comp['checks'],'computational_passed':comp['passed'],'computational_failed':comp['failed'],'wetlab_status':wet['status'],'OD600_values':wet['od_values'],'endpoint_cells':wet['endpoint_cells'],'training_replayed':False,'checkpoint_deserialized':False,'reserved_test_read':False}
(R/'verification_run.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
