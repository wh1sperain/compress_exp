"""Validate all original plans with their original immutable code release."""
import json,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
original=Path('/root/autodl-tmp/Compress/r3-releases/340eebd2e4c82bbd')
script='''import json
from pathlib import Path
from r3.runner import load_plan
from r3.common import read,digest
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
registration=root/'jobs/r3-20261024-batch-inference-v1/registration.json'
out=root/'raw/r3-20261024-batch-inference-v1'
bundle=dict(registration=str(registration),registration_sha256=digest(registration.read_bytes()),plans=[],reviewed_failure_hashes={},prefixes={},files={})
for item in read(registration)['plans']:
 p=Path(item['path']);assert digest(p.read_bytes())==item['sha256']
 plan,rows,identity=load_plan(p,root)
 bundle['plans'].append(dict(item=item,plan=plan,rows=rows,identity=identity))
 for name in ['spec_path','manifest_path','engine_profile_path','execution_acceptance_path','supplemental_audit_path','supplemental_status_path']:
  if plan.get(name):
   f=Path(plan[name]);bundle['files'][str(f)]=digest(f.read_bytes())
 for name in ['source_lock.json','model_weights_lock.json']:
  f=root/'manifests'/name;bundle['files'][str(f)]=digest(f.read_bytes())
 directory=out/p.stem
 for f in directory.glob('*/failure-*.json'):
  assert read(f)['type']=='KeyboardInterrupt', 'Unreviewed failure'
  bundle['reviewed_failure_hashes'][str(f)]=digest(f.read_bytes())
 for f in directory.glob('*/prefix.json'):
  if not (f.parent/'result.json').exists():bundle['prefixes'][str(f)]=dict(sha256=digest(f.read_bytes()),count=read(f)['generated_steps'])
print(json.dumps(bundle))
'''
r=subprocess.run(['/root/miniconda3/bin/python','-c',script],cwd=original,capture_output=True,text=True,check=True)
bundle=json.loads(r.stdout)
target=root/'raw/r3-20261024-batch-inference-v1'/('recovery-source-'+str(time.time_ns())+'.json')
with target.open('x') as f:json.dump(bundle,f)
print(json.dumps(dict(path=str(target),plans=len(bundle['plans']),reviewed_failures=len(bundle['reviewed_failure_hashes']),prefixes=bundle['prefixes'])))
