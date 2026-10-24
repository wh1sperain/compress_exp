import json
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
launch=json.loads((root/'telemetry/recovery_checks_launch.json').read_text());p=Path(launch['attempt'])
result={'launch':launch}
for file in [p/'state.json',*p.glob('*/receipt.json'),*p.glob('*/failure.json')]:
 if file.exists():result[str(file.relative_to(p))]=json.loads(file.read_text())
state=result.get('state.json',{})
if state.get('stage')=='failed':
 log=p/(state['current']+'.log');result['tail']=log.read_text(errors='replace')[-3000:]
print(json.dumps(result))
