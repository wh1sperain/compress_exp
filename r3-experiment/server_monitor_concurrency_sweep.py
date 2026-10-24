import json
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
launch=json.loads((root/'telemetry/concurrency_sweep_launch.json').read_text());p=Path(launch['attempt'])
r=dict(launch=launch)
for f in [p/'state.json',*p.glob('*/receipt.json'),*p.glob('*/failure.json')]:
 if f.exists():r[str(f.relative_to(p))]=json.loads(f.read_text())
for f in p.glob('*/failure.json'):
 log=p/(f.parent.name+'.log');r[f.parent.name+'_log_tail']=log.read_text(errors='replace')[-1700:]
print(json.dumps(r))
