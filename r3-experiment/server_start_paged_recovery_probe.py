"""Launch bounded candidate diagnostics only, never resume formal jobs."""
import json, os, subprocess, time
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
release=Path('/root/autodl-tmp/Compress/r3-releases/8d7abae5955b2de3')
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
assert not gpu.stdout.strip(), 'GPU occupied'
attempt=root/'tests'/('paged-recovery-'+str(time.time_ns()));attempt.mkdir()
driver='''import sys,json,time,os
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/Compress/r3-releases/8d7abae5955b2de3')
sys.path.append('/root/autodl-tmp/Compress/.venv-vllm/lib/python3.12/site-packages')
from r3 import qwen_batch_decode,qwen_paged_decode
from r3.batch_backend_probe import run
qwen_batch_decode.Backend=qwen_paged_decode.Backend
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
attempt=Path(__file__).parent
state=dict(stage='running',pid=os.getpid(),production_started=False,results=[])
def save():
 state['updated_at']=time.time();(attempt/'state.json').write_text(json.dumps(state))
save()
try:
 for condition in ['PriorTR25']:
  state['condition']=condition;save()
  result=run(root,'qwen8',condition,attempt/condition,steps=32,batch_size=3,benchmark=True)
  if result['status']!='PASS':raise RuntimeError('Numeric diagnostic failed')
  state['results'].append(condition);save()
 state.update(stage='complete',exit_code=0);save()
except BaseException as e:
 state.update(stage='failed',error=type(e).__name__+': '+str(e),exit_code=1);save();raise
'''
(attempt/'driver.py').write_text(driver)
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
env['PATH']='/root/autodl-tmp/Compress/.venv-vllm/bin:/usr/local/cuda/bin:'+env.get('PATH','')
with (attempt/'driver.log').open('xb') as log:
 child=subprocess.Popen([str(root/'environment/native-qwen/bin/python'),'-u',str(attempt/'driver.py')],cwd=release,env=env,stdout=log,stderr=log,start_new_session=True)
receipt=dict(pid=child.pid,attempt=str(attempt),release=str(release),production_started=False)
(root/'telemetry/paged_recovery_launch.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))



