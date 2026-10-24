"""One bounded read-only check of the formal inference lifecycle."""
import json,time,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
launch=json.loads((root/'telemetry/formal_inference_launch.json').read_text())
out=Path(launch['output']);result={'checked_at':time.time(),'launch':launch,'receipts':{},'workers':{}}
for name in ('lifecycle.json','status.json','completion.json','lifecycle_completion.json'):
    path=out/name
    if path.exists():result['receipts'][name]=json.loads(path.read_text())
for path in sorted(out.glob('*/status.json')):
    state=json.loads(path.read_text());state['result_files']=len(list(path.parent.glob('*/result.json')))
    key=state.get('job_key');prefix=path.parent/str(key)/'prefix.json'
    if key and prefix.exists():
        r=json.loads(prefix.read_text());state['active_generated_steps']=r['generated_steps'];state['prefix_updated_at']=r.get('updated_at')
    result['workers'][path.parent.name]=state
result['completed']=sum(x['result_files'] for x in result['workers'].values())
pids={launch['pid']}
for value in result['receipts'].values():
    pids.update(value[k] for k in ('pid','child_pid') if value.get(k))
result['processes_alive']={str(pid):Path('/proc/'+str(pid)).exists() for pid in pids}
for name in ('lifecycle-launch.log','coordinator.log'):
    path=out/name
    if path.exists():
        with path.open('rb') as f:f.seek(max(0,path.stat().st_size-1500));result[name+'_tail']=f.read().decode(errors='replace')
state=result['receipts'].get('status.json',{})
if state.get('current'):
    logs=sorted(out.glob(state['current']+'-*.log'),key=lambda p:p.stat().st_mtime)
    if logs:
        with logs[-1].open('rb') as f:f.seek(max(0,logs[-1].stat().st_size-2000));result['current_worker_log_tail']=f.read().decode(errors='replace')
export=Path(launch['archive_dir'])/'export_receipt.json'
if export.exists():result['export_receipt']=json.loads(export.read_text())
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
result['disk']=subprocess.run(['df','-h',str(root)],capture_output=True,text=True).stdout.strip()
print(json.dumps(result,ensure_ascii=False))
