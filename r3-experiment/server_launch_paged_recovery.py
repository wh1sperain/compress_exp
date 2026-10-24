"""Single reviewed continuation; immutable old successes and pause evidence retained."""
import json,os,signal,subprocess,time,hashlib
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3')
release=Path('/root/autodl-tmp/Compress/r3-releases/fb60dddd6256713c')
out=root/'raw/r3-20261024-batch-inference-v1'
source=out/'recovery-source-1792849546943052927.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (root/'telemetry/formal_recovery_launch.json').exists(),'Recovery already launched'
checks=json.loads((root/'telemetry/recovery_checks_launch.json').read_text())
state=json.loads((Path(checks['attempt'])/'state.json').read_text())
assert state['stage']=='complete' and state['exit_code']==0,'Checks incomplete'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(),'GPU occupied'
# Original workers already flushed checkpoints and exited. Retire only the two
# verified stopped parent controllers; never signal a live inference worker.
retired=[]
for pid,module in [(42345,'r3.formal_lifecycle'),(42346,'r3.run_coordinator')]:
 proc=Path('/proc')/str(pid)
 if proc.exists():
  cmd=(proc/'cmdline').read_bytes().decode().replace('\x00',' ')
  status=(proc/'status').read_text()
  assert module in cmd and 'r3-20261024-batch-inference-v1' in cmd and '\nState:\tT' in status,'Unexpected controller identity/state'
  os.kill(pid,signal.SIGKILL);retired.append(dict(pid=pid,module=module,previous_state='stopped'))
code=subprocess.check_output(['/root/miniconda3/bin/python','-c','from r3.common import code_hash;print(code_hash())'],cwd=release,text=True).strip()
rid='paged-recovery-'+str(time.time_ns())
profile=dict(backend='qwen-paged-recovery-v1',max_live_requests=4,max_live_requests_by_model=dict(llava7=4,mini8=4,qwen8=4,qwen4=4),
 dtype='bfloat16',kv_dtype='bfloat16',kv_scratch_factor=3,paged_cache_gib=24,prefix_replay='original_tokens',
 recovery_id=rid,recovery_code_sha256=code,parsing_enabled=False,scoring_enabled=False)
receipt=dict(folder=str(root),output=str(out),archive_dir=str(root/'exports/r3-20261024-batch-inference-v1'),
 source=str(source),source_sha256=sha(source),code_sha256=code,release=str(release),profile=profile,
 authorization='User requested complete remaining work and resume experiment after performance optimization',
 retired_stopped_controllers=retired,checks=checks,created_at=time.time(),full_native_engine_certification=False)
path=out/(rid+'.json')
with path.open('x') as f:json.dump(receipt,f)
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
env['PATH']='/root/autodl-tmp/Compress/.venv-vllm/bin:/usr/local/cuda/bin:'+env.get('PATH','')
with (out/(rid+'.log')).open('xb') as log:
 child=subprocess.Popen(['/root/miniconda3/bin/python','-u','-m','r3.paged_recovery','--receipt',str(path)],cwd=release,env=env,stdout=log,stderr=log,start_new_session=True)
launch=dict(pid=child.pid,receipt=str(path),release=str(release),started_at=time.time(),production_started=True)
with (root/'telemetry/formal_recovery_launch.json').open('x') as f:json.dump(launch,f)
print(json.dumps(launch))

