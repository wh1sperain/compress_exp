import json
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
r=json.loads((root/'telemetry/highres_capacity_launch.json').read_text());p=Path(r['attempt'])
s=json.loads((p/'state.json').read_text());r['state']=s
if s['stage']=='failed':r['log_tail']=(p/'driver.log').read_text(errors='replace')[-2500:]
print(json.dumps(r))
