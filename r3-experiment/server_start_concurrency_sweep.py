import json,os,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
release=Path('/root/autodl-tmp/Compress/r3-releases/0acd866fcaeb62a7')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
attempt=root/'tests'/('concurrency-sweep-'+str(time.time_ns()));attempt.mkdir()
driver='''import os,json,time,subprocess
from pathlib import Path
p=Path(__file__).parent;root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
state=dict(stage='running',pid=os.getpid(),results=[],production_started=False)
def save():state['updated_at']=time.time();(p/'state.json').write_text(json.dumps(state))
save()
for batch in [4,6,8,10]:
 with (p/(str(batch)+'.log')).open('xb') as log:
  child=subprocess.Popen([str(root/'environment/native-qwen/bin/python'),'-u','-m','r3.concurrency_probe','--folder',str(root),'--out',str(p/str(batch)),'--batch',str(batch)],stdout=log,stderr=log)
  state.update(current=batch,child_pid=child.pid);save();rc=child.wait()
 state['results'].append(dict(batch=batch,exit_code=rc));save()
 if rc:break
 receipts=[json.loads(x.read_text()) for x in p.glob('*/receipt.json')]
 if len(receipts)>1:
  ordered=sorted(receipts,key=lambda x:x['batch']);last,prev=ordered[-1],ordered[-2]
  ratio=last['scenarios']['mixed']['tokens_per_second']/prev['scenarios']['mixed']['tokens_per_second']
  if ratio<1.05:break
receipts=[json.loads(x.read_text()) for x in p.glob('*/receipt.json')]
if not receipts:state.update(stage='failed',exit_code=1);save();raise SystemExit(1)
best=max(receipts,key=lambda x:x['scenarios']['mixed']['tokens_per_second'])
state.update(stage='complete',exit_code=0,selected_batch=best['batch'],cache_gib=best['cache_gib'],selected_receipt=str(p/str(best['batch'])/'receipt.json'));save()
'''
(attempt/'driver.py').write_text(driver)
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
env['PATH']='/root/autodl-tmp/Compress/.venv-vllm/bin:/usr/local/cuda/bin:'+env.get('PATH','')
with (attempt/'driver.log').open('xb') as log:
 child=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(attempt/'driver.py')],cwd=release,env=env,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=child.pid,attempt=str(attempt),release=str(release),production_started=False)
(root/'telemetry/concurrency_sweep_launch.json').write_text(json.dumps(r));print(json.dumps(r))
