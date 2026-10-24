import json,os,signal,subprocess,time,hashlib
from pathlib import Path
root=Path('/root/autodl-tmp/Compress/runs/visual-compression-r3');out=root/'raw/r3-20261024-batch-inference-v1'
release=Path('/root/autodl-tmp/Compress/r3-releases/0acd866fcaeb62a7')
source=out/'recovery-source-1792850683170330782.json'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
launch=read(root/'telemetry/concurrency_sweep_launch.json');test=read(Path(launch['attempt'])/'state.json')
assert test['stage']=='complete' and test['exit_code']==0
best=read(Path(test['selected_receipt']));assert best['status']=='PASS' and best['margin_bytes']>=8*1024**3
capacity_launch=read(root/'telemetry/highres_capacity_launch.json')
capacity=read(Path(capacity_launch['attempt'])/'state.json')
assert capacity.get('status')=='PASS' and capacity.get('batch')==best['batch'],'High-resolution compressed capacity not passed'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
assert not (root/'telemetry/formal_tuned_launch.json').exists(),'Already resumed tuning'
prior=read(out/'recovery_status.json');pid=prior['pid'];proc=Path('/proc')/str(pid)
assert 'r3.paged_recovery' in (proc/'cmdline').read_bytes().decode() and '\nState:\tT' in (proc/'status').read_text()
os.kill(pid,signal.SIGKILL)
rid='tuned-recovery-'+str(time.time_ns())
with (out/(rid+'-previous-state.json')).open('x') as f:json.dump(prior,f)
code=subprocess.check_output(['/root/miniconda3/bin/python','-c','from r3.common import code_hash;print(code_hash())'],cwd=release,text=True).strip()
profile=dict(backend='qwen-paged-recovery-v1',max_live_requests=best['batch'],max_live_requests_by_model=dict(llava7=4,mini8=4,qwen8=best['batch'],qwen4=best['batch']),
 dtype='bfloat16',kv_dtype='bfloat16',kv_scratch_factor=3,paged_cache_gib=best['cache_gib'],prefix_replay='original_tokens',recovery_id=rid,recovery_code_sha256=code,
 parsing_enabled=False,scoring_enabled=False)
receipt=dict(folder=str(root),output=str(out),archive_dir=str(root/'exports/r3-20261024-batch-inference-v1'),source=str(source),source_sha256=sha(source),code_sha256=code,release=str(release),profile=profile,
 authorization='User requested test 6 and progressively increase concurrency while speed improves, then complete at fastest measured stable speed',
 previous_coordinator=pid,created_at=time.time(),full_native_engine_certification=False,concurrency_test=launch,concurrency_result=best,high_resolution_capacity=capacity)
path=out/(rid+'.json')
with path.open('x') as f:json.dump(receipt,f)
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
env['PATH']='/root/autodl-tmp/Compress/.venv-vllm/bin:/usr/local/cuda/bin:'+env.get('PATH','')
with (out/(rid+'.log')).open('xb') as log:
 child=subprocess.Popen(['/root/miniconda3/bin/python','-u','-m','r3.paged_recovery','--receipt',str(path)],cwd=release,env=env,stdout=log,stderr=log,start_new_session=True)
r=dict(pid=child.pid,receipt=str(path),release=str(release),started_at=time.time(),production_started=True,batch=best['batch'],cache_gib=best['cache_gib'])
with (root/'telemetry/formal_tuned_launch.json').open('x') as f:json.dump(r,f)
print(json.dumps(r))
