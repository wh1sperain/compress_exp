import json,os,signal,time,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
out=root/'raw/r3-20261024-batch-inference-v1'
s=json.loads((out/'recovery_status.json').read_text());parent=s['pid'];worker=s['child_pid']
assert s['stage']=='running'
for pid in (parent,worker):
 assert 'r3.paged_recovery' in (Path('/proc')/str(pid)/'cmdline').read_bytes().decode().replace('\0',' ')
os.kill(parent,signal.SIGSTOP)
os.kill(worker,signal.SIGINT)
for _ in range(30):
 p=Path('/proc')/str(worker)
 if not p.exists() or '\nState:\tZ' in (p/'status').read_text():break
 time.sleep(.5)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(),'Worker has not released GPU'
w=out/s['current'];f=max(w.glob('worker-failure-*.json'),key=lambda x:x.stat().st_mtime)
failure=json.loads(f.read_text());assert failure['type']=='KeyboardInterrupt'
prefixes={k:json.loads((w/k/'prefix.json').read_text())['generated_steps'] for k in failure['started_unfinished'] if (w/k/'prefix.json').exists()}
r=dict(time=time.time(),coordinator=parent,worker=worker,scope=s['current'],prefix_lengths=prefixes,completed=json.loads((w/'status.json').read_text())['completed'],reason='User explicitly requested progressive concurrency tuning for fastest completion')
(out/('tuning-pause-'+str(time.time_ns())+'.json')).write_text(json.dumps(r));print(json.dumps(r))
