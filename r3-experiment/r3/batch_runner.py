"""Bound-plan single-model batch worker with immutable raw result journals."""
import argparse
import os
import time
from pathlib import Path
from .common import Blocked, read, write
from .runner import load_plan, supported, publish_once, canonical


def validate_profile(profile, expected_max_live=None):
    if profile.get('backend') != 'native-padded-batch':
        raise Blocked('Expected frozen native-padded-batch profile')
    slots=profile.get('max_live_requests')
    if type(slots) is not int or not 1<=slots<=32:
        raise Blocked('Invalid approved batch slot count')
    if expected_max_live is not None and slots!=expected_max_live:
        raise Blocked('Profile differs from expected live slot count')
    if profile.get('dtype')!='bfloat16' or profile.get('kv_dtype')!='bfloat16':
        raise Blocked('Batch precision must remain BF16')


def validate_generation(generated, job):
    ids=generated.get('generated_token_ids');cap=job['generation']['max_new_tokens']
    if not isinstance(ids,list) or not ids or any(type(token) is not int or token<0 for token in ids):
        raise Blocked('Missing or malformed raw token IDs')
    if len(ids)!=generated.get('output_tokens') or len(ids)>cap:
        raise Blocked('Invalid raw output count')
    if not isinstance(generated.get('output_text'),str):raise Blocked('Missing full raw output text')
    if generated.get('output_cap_reached')!=(len(ids)==cap):raise Blocked('Incorrect cap flag')
    eos=generated.get('generation_eos_token_id')
    eos=[] if eos is None else [eos] if isinstance(eos,int) else eos
    if not isinstance(eos,list) or not eos:raise Blocked('Native EOS identity missing')
    if generated.get('eos_terminated')!=(ids[-1] in eos):raise Blocked('Incorrect EOS flag')
    if any(token in eos for token in ids[:-1]):raise Blocked('Generation continued after EOS')
    if not generated['eos_terminated'] and len(ids)!=cap:raise Blocked('Unexpected early termination')


def run_jobs(jobs,rows,identity,out,profile,session_factory,scheduler_factory,expected_max_live=None):
    """Factories allow CPU journal tests; the CLI validates plan and holds lock."""
    validate_profile(profile,expected_max_live)
    if not identity.get('inference_only'):raise Blocked('Batch raw worker requires inference-only policy')
    if not jobs or any(not supported(job) for job in jobs):raise Blocked('Unsupported registered jobs')
    scopes={(job['model'],job['condition']) for job in jobs}
    if len(scopes)!=1:raise Blocked('Exactly one model and condition per resident worker')
    if len({j['job_key'] for j in jobs})!=len(jobs):raise Blocked('Duplicate plan job')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pending=[];bindings={};completed=0;started={}
    for job in jobs:
        key=job['job_key']
        if len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise Blocked('Invalid job key')
        row=rows[job['uid']]
        if row['task']!=job['task'] or row['image_sha256']!=job['image_sha256']:
            raise Blocked('Frozen input/job mismatch')
        directory=out/key;directory.mkdir(exist_ok=True)
        binding=dict(job=job,input_sha256=canonical(row),identity=identity)
        path=directory/'binding.json'
        if path.exists():
            if read(path)!=binding:raise Blocked('Original job/input/source binding changed')
        else:publish_once(path,binding)
        bindings[key]=binding
        result_path=directory/'result.json'
        if result_path.exists():
            saved=read(result_path)
            if saved.get('binding_sha256')!=canonical(binding) or saved.get('record_complete') is not True:
                raise Blocked('Invalid immutable success receipt')
            if any(saved.get(k)!=v for k,v in job.items()):raise Blocked('Existing result does not match registered job')
            validate_generation(saved,job);completed+=1;continue
        if list(directory.glob('failure-*.json')):
            raise Blocked('Prior failure requires reviewed recovery; prefix remains unchanged')
        pending.append((row,job,directory/'prefix.json'))
    def status(stage):
        write(out/'status.json',dict(stage=stage,completed=completed,total=len(jobs),
              pid=os.getpid(),updated_at=time.time(),parsing_enabled=False,scoring_enabled=False))
    def requests():
        for row,job,path in pending:
            # Mark before handing off, including prefill failures and any fetched
            # pending request. Never invent a result for a partially started job.
            started[job['job_key']]=job
            yield row,job,path
    if not pending:status('complete');return completed
    model,condition=next(iter(scopes))
    try:
        status('loading')
        session=session_factory(model,condition,out/('loading-'+str(time.time_ns())+'.json'))
        scheduler=scheduler_factory(session,profile)
        status('running')
        for job,row,generated in scheduler.run(requests(),out/'batch_status.json'):
            key=job['job_key']
            if key not in started or job!=started[key] or canonical(row)!=bindings[key]['input_sha256']:
                raise Blocked('Scheduler returned an unbound request')
            validate_generation(generated,job)
            result=dict(job,**generated,record_complete=True,binding_sha256=canonical(bindings[key]),
                completed_at=time.time(),scoring_status='disabled_by_user',parsing_status='disabled_by_user',
                full_native_engine_certification=False)
            if identity.get('execution_mode'):result['execution_mode']=identity['execution_mode']
            publish_once(out/key/'result.json',result)
            started.pop(key);completed+=1;status('running')
        if completed!=len(jobs) or started:raise Blocked('Scheduler returned before all registered jobs completed')
    except BaseException as exc:
        for key,job in started.items():
            if not (out/key/'result.json').exists():
                publish_once(out/key/('failure-'+str(time.time_ns())+'.json'),dict(job_key=key,
                    type=type(exc).__name__,message=str(exc)[:1200],binding_sha256=canonical(bindings[key]),
                    time=time.time(),resampled=False,original_prefix_preserved=True))
        publish_once(out/('worker-failure-'+str(time.time_ns())+'.json'),dict(type=type(exc).__name__,
            message=str(exc)[:1200],started_unfinished=list(started),completed=completed,resampled=False))
        status('failed');raise
    status('complete');return completed


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--out',required=True);p.add_argument('--expected-max-live',type=int);a=p.parse_args()
    folder=Path(a.folder)
    import fcntl
    with (folder/'production.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,rows,identity=load_plan(a.plan,folder)
        if plan.get('backend')!='native-padded-batch':raise Blocked('Batch runner requires registered batch plan')
        profile=read(plan['engine_profile_path'])
        from .native_session import NativeSession
        from .batch_scheduler import Scheduler
        run_jobs(plan['jobs'],rows,identity,a.out,profile,
                 lambda m,c,r:NativeSession(folder,m,c,r),Scheduler,a.expected_max_live)


if __name__=='__main__':main()
