import json,subprocess,os,time
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3');release=Path('/root/autodl-tmp/Compress/r3-releases/0acd866fcaeb62a7')
launch=json.loads((root/'telemetry/concurrency_sweep_launch.json').read_text());state=json.loads((Path(launch['attempt'])/'state.json').read_text());assert state['stage']=='complete'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
attempt=root/'tests'/('highres-capacity-'+str(time.time_ns()));attempt.mkdir()
driver='''import sys,json,time,os
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/Compress/r3-releases/0acd866fcaeb62a7')
sys.path.append('/root/autodl-tmp/Compress/.venv-vllm/lib/python3.12/site-packages')
import torch
from r3.native_session import NativeSession
from r3.qwen_paged_decode import Backend
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3');out=Path(__file__).parent
state=dict(stage='running',pid=os.getpid(),completed_prefills=0)
def save():state['updated_at']=time.time();(out/'state.json').write_text(json.dumps(state))
save()
try:
 rows=[json.loads(x) for x in (root/'manifests/frozen-inputs-v1/calibration_manifest.jsonl').read_text().splitlines()]
 rows=sorted(rows,key=lambda x:x['native_inputs']['qwen8']['expanded_prompt_tokens'],reverse=True)[:10]
 session=NativeSession(root,'qwen8','PriorTR25',out/'loading.json');session.paged_batch_size=10;session.paged_cache_gib=60
 backend=Backend(session);states=[];torch.cuda.reset_peak_memory_stats()
 for row in rows:
  states.append(backend.prefill(row,dict(job_key='capacity-'+row['uid'],generation=dict(max_new_tokens=32768))))
  state['completed_prefills']+=1;save()
 for _ in range(8):
  result=backend.decode(states,torch.full((10,),42,device=session.model.device,dtype=torch.long));assert torch.isfinite(result).all()
 peak=torch.cuda.max_memory_allocated();total=torch.cuda.mem_get_info()[1]
 assert total-peak>=8*1024**3,'Less than 8GiB margin'
 state.update(stage='complete',status='PASS',batch=10,cache_gib=60,peak_allocated_bytes=peak,margin_bytes=total-peak,max_expanded_prompt=max(r['native_inputs']['qwen8']['expanded_prompt_tokens'] for r in rows),uids=[r['uid'] for r in rows]);save()
except BaseException as e:
 state.update(stage='failed',status='FAIL',error_type=type(e).__name__,error=str(e)[:1200]);save();raise
'''
(attempt/'driver.py').write_text(driver)
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
env['PATH']='/root/autodl-tmp/Compress/.venv-vllm/bin:/usr/local/cuda/bin:'+env.get('PATH','')
with (attempt/'driver.log').open('xb') as log:
 child=subprocess.Popen([str(root/'environment/native-qwen/bin/python'),'-u',str(attempt/'driver.py')],cwd=release,env=env,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=child.pid,attempt=str(attempt));(root/'telemetry/highres_capacity_launch.json').write_text(json.dumps(r));print(json.dumps(r))
