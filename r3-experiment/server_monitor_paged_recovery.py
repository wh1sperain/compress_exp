import json,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
launch=json.loads((root/'telemetry/paged_recovery_launch.json').read_text())
attempt=Path(launch['attempt'])
out={'launch':launch}
for name in ['state.json','PriorTR25/performance.json','Full/status.json','Full/receipt.json','Full/failure.json','PriorTR25/status.json','PriorTR25/receipt.json','PriorTR25/failure.json']:
 p=attempt/name
 if p.exists():out[name]=json.loads(p.read_text())
p=attempt/'driver.log'
if p.exists():out['log_tail']=p.read_text(errors='replace')[-3500:]
out['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(out))

