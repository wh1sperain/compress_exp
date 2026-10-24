"""Reviewed backend migration; original plan/job bindings and successes stay intact."""
import argparse,os,subprocess,time
from pathlib import Path
from .common import read,write,digest,code_hash,Blocked
from .runner import publish_once,canonical

def load(receipt):
    r=read(receipt)
    if r['code_sha256']!=code_hash():raise Blocked('Recovery code changed')
    source=Path(r['source'])
    if digest(source.read_bytes())!=r['source_sha256']:raise Blocked('Recovery source changed')
    b=read(source)
    if digest(Path(b['registration']).read_bytes())!=b['registration_sha256']:raise Blocked('Registration changed')
    for path,sha in b['files'].items():
        if digest(Path(path).read_bytes())!=sha:raise Blocked('Frozen input/source changed: '+path)
    for entry in b['plans']:
        if digest(Path(entry['item']['path']).read_bytes())!=entry['item']['sha256']:raise Blocked('Plan changed')
    return r,b

def worker(receipt,index):
    import fcntl
    r,b=load(receipt);root=Path(r['folder']);entry=b['plans'][index]
    with (root/'production.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from .native_session import NativeSession
        from .batch_scheduler import Scheduler
        from .batch_runner import run_jobs
        profile=dict(r['profile'],reviewed_failure_hashes=b['reviewed_failure_hashes'])
        if not entry['plan']['jobs'][0]['model'].startswith('qwen'):
            profile['backend']='native-padded-batch'
        run_jobs(entry['plan']['jobs'],entry['rows'],entry['identity'],
                 Path(r['output'])/Path(entry['item']['path']).stem,profile,
                 lambda m,c,p:NativeSession(root,m,c,p),Scheduler)

def coordinate(receipt):
    import fcntl
    r,b=load(receipt);root=Path(r['folder']);out=Path(r['output'])
    with (root/'production-coordinator.lock').open('a') as lock, (root/'formal-lifecycle.lock').open('a') as lifecycle:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lifecycle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path,item in b['prefixes'].items():
            if digest(Path(path).read_bytes())!=item['sha256']:raise Blocked('Original prefix changed before recovery')
        state=dict(pid=os.getpid(),stage='running',recovery_id=r['profile']['recovery_id'],results=[],started_at=time.time(),parsing_enabled=False,scoring_enabled=False)
        def save():state['updated_at']=time.time();write(out/'recovery_status.json',state)
        save()
        try:
            for index,entry in enumerate(b['plans']):
                item=entry['item'];name=Path(item['path']).stem
                # Workers verify immutable existing successes even for completed arms.
                command=[str(root/('environment/native-'+item['environment']+'/bin/python')),'-u','-m','r3.paged_recovery','--receipt',str(receipt),'--worker-index',str(index)]
                with (out/(name+'-recovery-'+str(time.time_ns())+'.log')).open('xb') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=log)
                    state.update(current=name,child_pid=child.pid);save();rc=child.wait()
                state['results'].append(dict(plan=name,exit_code=rc));save()
                if rc:raise Blocked('Recovery worker failed; no automatic retry')
            state.update(stage='generation_complete',exit_code=0);save()
            publish_once(out/'completion.json',state)
            from .export_results import export
            state.update(stage='exporting');save()
            exported=export(root,b['registration'],out,r['archive_dir'],snapshot=False)
            state.update(stage='complete',completed_at=time.time(),export_receipt=exported);save()
            publish_once(out/'lifecycle_completion.json',state)
        except BaseException as e:
            state.update(stage='failed',error_type=type(e).__name__,error=str(e)[:1000]);save()
            publish_once(out/('recovery_failure-'+str(time.time_ns())+'.json'),state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--worker-index',type=int);a=p.parse_args()
    # Only append the already installed previous runtime's FlashInfer dependencies.
    import sys
    sys.path.append('/root/autodl-tmp/Compress/.venv-vllm/lib/python3.12/site-packages')
    if a.worker_index is None:coordinate(a.receipt)
    else:worker(a.receipt,a.worker_index)
