import json,os,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
release=Path('/root/autodl-tmp/Compress/r3-releases/36ddbc67a2b57c33')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
attempt=root/'tests'/('recovery-integration-'+str(time.time_ns()));attempt.mkdir()
driver='''import sys,subprocess,json,os,time
from pathlib import Path
attempt=Path(__file__).parent
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
release='/root/autodl-tmp/Compress/r3-releases/36ddbc67a2b57c33'
state=dict(stage='running',pid=os.getpid(),results=[],production_started=False)
def save():state['updated_at']=time.time();(attempt/'state.json').write_text(json.dumps(state))
base="import sys;sys.path.insert(0,"+repr(release)+");sys.path.append('/root/autodl-tmp/Compress/.venv-vllm/lib/python3.12/site-packages');"
cases=[('slots',"from r3.paged_slot_probe import run;run("+repr(str(root))+","+repr(str(attempt/'slots'))+")")]
for model,cond in [('qwen8','PriorTR50'),('qwen4','PriorTR25')]:
 name=model+'-'+cond
 code="from r3 import qwen_batch_decode,qwen_paged_decode;qwen_batch_decode.Backend=qwen_paged_decode.Backend;from r3.batch_backend_probe import run;r=run("+repr(str(root))+","+repr(model)+","+repr(cond)+","+repr(str(attempt/name))+",steps=16,batch_size=3);assert r['status']=='PASS'"
 cases.append((name,code))
save()
for name,code in cases:
 with (attempt/(name+'.log')).open('xb') as log:
  p=subprocess.Popen([str(root/'environment/native-qwen/bin/python'),'-u','-c',base+code],stdout=log,stderr=log);state.update(current=name,child_pid=p.pid);save();rc=p.wait()
 state['results'].append(dict(name=name,exit_code=rc));save()
 if rc:state.update(stage='failed',exit_code=rc);save();raise SystemExit(rc)
state.update(stage='complete',exit_code=0);save()
'''
(attempt/'driver.py').write_text(driver)
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
env['PATH']='/root/autodl-tmp/Compress/.venv-vllm/bin:/usr/local/cuda/bin:'+env.get('PATH','')
with (attempt/'driver.log').open('xb') as log:
 child=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(attempt/'driver.py')],cwd=release,env=env,stdout=log,stderr=log,start_new_session=True)
receipt=dict(pid=child.pid,attempt=str(attempt),release=str(release),production_started=False)
(root/'telemetry/recovery_checks_launch.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
