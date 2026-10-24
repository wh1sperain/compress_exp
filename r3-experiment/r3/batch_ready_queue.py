"""Bounded checks for the new scheduler, then authorized formal inference.

Previous numeric evidence is reused. No answers are parsed or scored. Any failed
check preserves its receipt and prevents the automatic formal handoff.
"""
import argparse,os,subprocess,time
from pathlib import Path
from .common import read,write,digest,code_hash,Blocked
from .runner import publish_once

def run(folder):
    folder=Path(folder);release=Path(__file__).resolve().parent.parent
    import fcntl
    with (folder/'batch-ready.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=folder/'tests'/('batch-ready-'+str(time.time_ns()));attempt.mkdir(parents=True)
        state=dict(pid=os.getpid(),stage='running',attempt=str(attempt),release=str(release),results=[],production_started=False)
        def update():
            state['updated_at']=time.time();write(attempt/'state.json',state);write(folder/'telemetry/batch_ready_state.json',state)
        update()
        proof=[];numeric=[]
        source=Path('/root/autodl-tmp/Compress/r3-releases/4f0dbe63ebd03d2f')
        previous=folder/'tests/batch-backend-1792829876132699351'
        for name in ('llava7-Full','llava7-FastV25','llava7-PriorTR25','qwen8-Full','qwen8-PriorTR25','mini8-MiniNative'):
            path=previous/name/'receipt.json';r=read(path)
            if r.get('status')!='PASS':raise Blocked('Prior actual batch probe did not pass')
            numeric.append(dict(path=str(path),sha256=digest(path.read_bytes()),source_release=str(source),source_code_sha256=r['code_sha256']))
        checks=[
            ('llava','numeric-Agile25','r3.batch_backend_probe',['--model','llava7','--condition','AgilePruner25','--steps','32','--batch-size','4']),
            ('llava','numeric-VisionZip25','r3.batch_backend_probe',['--model','llava7','--condition','VisionZip25','--steps','32','--batch-size','4']),
            ('qwen','numeric-Qwen4-Prior25','r3.batch_backend_probe',['--model','qwen4','--condition','PriorTR25','--steps','32','--batch-size','4']),
            ('llava','memory-llava','r3.batch_resource_probe',['--model','llava7','--condition','Full','--batch-size','4']),
            ('qwen','memory-qwen','r3.batch_resource_probe',['--model','qwen8','--condition','Full','--batch-size','4']),
            ('mini','memory-mini','r3.batch_resource_probe',['--model','mini8','--condition','MiniNative','--batch-size','4']),
            ('llava','slots-llava','r3.batch_scheduler_smoke',['--model','llava7','--condition','Full']),
            ('mini','slots-mini','r3.batch_scheduler_smoke',['--model','mini8','--condition','MiniNative']),
            ('qwen','slots-qwen','r3.batch_scheduler_smoke',['--model','qwen8','--condition','Full']),
        ]
        try:
            for environment,name,module,args in checks:
                destination=attempt/name
                command=[str(folder/('environment/native-'+environment+'/bin/python')),'-u','-m',module,
                         '--folder',str(folder),'--out',str(destination),*args]
                with (attempt/(name+'.log')).open('xb') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=log)
                    state.update(current=name,child_pid=child.pid);update();rc=child.wait()
                state['results'].append(dict(name=name,exit_code=rc));update()
                if rc:raise Blocked('New batch check failed: '+name)
                path=destination/'receipt.json';r=read(path)
                if r.get('status')!='PASS':raise Blocked('New batch check has no PASS: '+name)
                item=dict(path=str(path),sha256=digest(path.read_bytes()),source_release=str(release),source_code_sha256=r['code_sha256'])
                if module=='r3.batch_backend_probe':numeric.append(item)
                else:proof.append(item)
            from .execution_acceptance import BATCH_FILES,validate_batch_profile
            profile=dict(backend='native-padded-batch',dtype='bfloat16',kv_dtype='bfloat16',max_live_requests=4,
                kv_scratch_factor=3,memory_admission='full_remaining_cap_with_scratch',persistent_model=True,
                parallel_model_copies=False,continuous_slot_refill=True,paged_kv=False,cuda_graph=False,
                prefill_policy='complete per-request author native prefill',parsing_enabled=False,scoring_enabled=False,
                checkpoint_interval_tokens=128,backend_impl_sha256={name:digest((release/name).read_bytes()) for name in BATCH_FILES},
                batch_probe_receipts=numeric,execution_check_receipts=proof,full_native_engine_certification=False,
                timing_limitation='Short fixed-prefix timings are diagnostic; full-run throughput measured separately; Mini first-call network timing excluded from speed claims')
            validate_batch_profile(profile)
            profile_path=attempt/'approved_profile.json';publish_once(profile_path,profile)
            state.update(stage='launching_formal',approved_profile=str(profile_path));update()
            from .start_formal import run as start
            launch=start(folder,profile_path)
            state.update(stage='formal_started',production_started=True,formal_launch=launch,exit_code=0);update()
            publish_once(attempt/'completion.json',state)
        except BaseException as exc:
            state.update(stage='failed',error_type=type(exc).__name__,error=str(exc)[:1200],exit_code=1);update()
            publish_once(attempt/('failure-'+str(time.time_ns())+'.json'),state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();run(a.folder)
