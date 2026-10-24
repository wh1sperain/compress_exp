"""Gate-locked persistent B1 worker and immutable per-job result journal.

This executable never manufactures a certification receipt. An unsupported arm,
stale gate, changed image/input, or failed job stops execution without resampling.
"""
import argparse
import json
import os
import time
from pathlib import Path
from .common import read, write, digest, code_hash, require_gate, Blocked


def canonical(value):return digest(json.dumps(value,sort_keys=True,ensure_ascii=False))


def publish_once(path,value):
    """Atomic no-replace publication on the same filesystem, with fsync."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n').encode()
    temporary=path.with_name(path.name+'.pending-'+str(os.getpid()))
    with temporary.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    try:
        # A hard link atomically fails if a previous successful result exists.
        os.link(temporary,path)
    finally:temporary.unlink()


def load_plan(plan_path,folder):
    plan=read(plan_path);folder=Path(folder)
    if plan.get('backend')!='native-b1':raise Blocked('Unimplemented backend')
    spec_path=Path(plan['spec_path']);spec_sha=digest(spec_path.read_bytes())
    manifest=Path(plan['manifest_path']);manifest_sha=digest(manifest.read_bytes())
    profile=Path(plan['engine_profile_path']);profile_sha=digest(profile.read_bytes())
    identity={'manifest_sha256':manifest_sha,
              'source_lock_sha256':digest((folder/'manifests/source_lock.json').read_bytes()),
              'weights_lock_sha256':digest((folder/'manifests/model_weights_lock.json').read_bytes())}
    gates=[]
    for name in ('native_gate_path','engine_gate_path'):
        gate=require_gate(plan[name],spec_sha,profile_sha)
        gates.append(gate)
        if any(gate.get(k)!=v for k,v in identity.items()):raise Blocked('Gate input/source lock changed')
    if plan.get('spec_sha256')!=spec_sha or plan.get('engine_sha256')!=profile_sha:
        raise Blocked('Plan spec/profile changed')
    if plan.get('code_sha256')!=code_hash():raise Blocked('Plan code changed')
    rows=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
    by_uid={r['uid']:r for r in rows}
    if len(by_uid)!=len(rows):raise Blocked('Duplicate manifest uid')
    spec=read(spec_path)
    # Rebuild the complete registered main/controls set instead of trusting
    # arbitrary generation settings in an edited plan.
    from .build_jobs import build
    expected,_=build(spec,rows,None,profile_sha)
    expected={j['job_key']:j for j in expected}
    if not plan.get('jobs'):raise Blocked('Empty plan')
    if len({j['job_key'] for j in plan['jobs']})!=len(plan['jobs']):raise Blocked('Duplicate registered job')
    for job in plan['jobs']:
        if job!=expected.get(job['job_key']):raise Blocked('Unregistered or modified job')
        scope=job['model']+'/'+job['condition'].removesuffix('_T0p6')
        if any(scope not in gate.get('certified_conditions',[]) for gate in gates):
            raise Blocked('Condition outside gate certification scope')
    return plan,by_uid,dict(identity,spec_sha256=spec_sha,engine_sha256=profile_sha,code_sha256=code_hash())


def supported(job):
    condition=job['condition'].removesuffix('_T0p6')
    allowed={'llava7':{'Full','FastV50','FastV25','PriorTR50','PriorTR25','Agile50','Agile25','VisionZip50','VisionZip25'},
             'qwen8':{'Full','PriorTR50','PriorTR25'},'mini8':{'MiniNative'}}
    return condition in allowed.get(job['model'],set()) and job['stage'] in ('main','controls')


def run_jobs(jobs,rows,identity,out,session_factory):
    """Testable worker core; CLI acquires an exclusive lock and validates gates."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not jobs or any(not supported(j) for j in jobs):raise Blocked('Unsupported condition; no jobs started')
    pairs={(j['model'],j['condition']) for j in jobs}
    if len(pairs)!=1:raise Blocked('One frozen model/condition per isolated worker')
    model,condition=next(iter(pairs));session=None;completed=0
    for job in jobs:
        key=job['job_key']
        if len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise Blocked('Invalid job key')
        row=rows[job['uid']]
        if row['image_sha256']!=job['image_sha256'] or row['task']!=job['task']:raise Blocked('Manifest/job mismatch')
        directory=out/key;directory.mkdir(exist_ok=True)
        binding={'job':job,'input_sha256':canonical(row),'identity':identity}
        binding_path=directory/'binding.json'
        if binding_path.exists():
            if read(binding_path)!=binding:raise Blocked('Original job/input/source binding changed')
        else:publish_once(binding_path,binding)
        result_path=directory/'result.json'
        if result_path.exists():
            result=read(result_path)
            if result.get('binding_sha256')!=canonical(binding) or not result.get('record_complete'):
                raise Blocked('Invalid existing success receipt')
            completed+=1;continue
        failures=list(directory.glob('failure-*.json'))
        if failures:raise Blocked('Failed job requires explicit validated recovery; original prefix retained')
        try:
            if session is None:session=session_factory(model,condition,out/('loading-'+str(time.time_ns())+'.json'))
            write(out/'status.json',{'stage':'running','completed':completed,'total':len(jobs),
                  'job_key':key,'pid':os.getpid(),'updated_at':time.time()})
            generated=session.generate(row,job,directory/'prefix.json')
            ids=generated['generated_token_ids'];count=len(ids);cap=job['generation']['max_new_tokens']
            if count!=generated['output_tokens'] or not 0<count<=cap:raise Blocked('Invalid output count')
            if generated['output_cap_reached']!=(count==cap):raise Blocked('Incorrect cap flag')
            if not generated['eos_terminated'] and count!=cap:raise Blocked('Unexpected early termination')
            result=dict(job,**generated,record_complete=True,binding_sha256=canonical(binding),
                        completed_at=time.time(),scoring_status='pending_separate_label_process')
            publish_once(result_path,result)
            completed+=1
        except BaseException as exc:
            publish_once(directory/('failure-'+str(time.time_ns())+'.json'),
                {'job_key':key,'type':type(exc).__name__,'message':str(exc)[:1200],
                 'binding_sha256':canonical(binding),'time':time.time(),'resampled':False})
            write(out/'status.json',{'stage':'failed','completed':completed,'job_key':key,'updated_at':time.time()})
            raise
    write(out/'status.json',{'stage':'complete','completed':completed,'total':len(jobs),'updated_at':time.time()})
    return completed


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    folder=Path(a.folder);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (folder/'production.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,rows,identity=load_plan(a.plan,folder)
        from .native_session import NativeSession
        run_jobs(plan['jobs'],rows,identity,out,
                 lambda m,c,r:NativeSession(folder,m,c,r))


if __name__=='__main__':main()
