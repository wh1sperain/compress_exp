"""Calibration-only candidate worker; no formal jobs and no fabricated gate."""
import argparse
import json
import os
import time
from pathlib import Path
from .common import Blocked, read, write, digest, code_hash
from .runner import canonical, publish_once
from .natural_preparation import build, CONDITIONS


def plan(folder,model,condition):
    folder=Path(folder)
    manifest=folder/'manifests/frozen-inputs-v1/calibration_manifest.jsonl'
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    lock=read(manifest.parent/'manifest_lock.json')
    # Frozen filename alone is insufficient: verify against its lock values.
    if digest(manifest.read_bytes()) != lock.get('hashes',{}).get(manifest.name):raise Blocked('Calibration lock mismatch')
    extra={}
    if model=='qwen4':
        from .supplemental_inputs import attach
        audit=read(folder/'telemetry/qwen4_supplemental_audit_launch.json')
        rows,extra=attach(rows,audit['out'],audit['status_path'])
    spec_path=Path(__file__).resolve().parent.parent/'round3_spec.json'
    matrix=build(read(spec_path),rows,digest(manifest.read_bytes()),include_qwen4=model=='qwen4')
    if condition not in CONDITIONS.get(model,[]):raise Blocked('Unregistered calibration condition')
    jobs=[j for j in matrix['jobs'] if j['model']==model and j['condition']==condition and j['side']=='candidate']
    identity=dict(manifest_sha256=digest(manifest.read_bytes()),spec_sha256=digest(spec_path.read_bytes()),
        code_sha256=code_hash(),source_lock_sha256=digest((folder/'manifests/source_lock.json').read_bytes()),
        weights_lock_sha256=digest((folder/'manifests/model_weights_lock.json').read_bytes()),**extra)
    return jobs,{r['uid']:r for r in rows},identity


def run(folder,model,condition,out,integration_smoke=False):
    import fcntl
    from .native_session import NativeSession
    folder,out=Path(folder),Path(out)
    jobs,rows,identity=plan(folder,model,condition)
    out.mkdir(parents=True,exist_ok=True)
    with (folder/'correctness.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding=dict(model=model,condition=condition,side='candidate',jobs=jobs,identity=identity)
        if (out/'plan.json').exists():
            if read(out/'plan.json')!=binding:raise Blocked('Calibration binding changed')
        else:publish_once(out/'plan.json',binding)
        session=None;done=0;started=time.time()
        state=dict(pid=os.getpid(),stage='running',model=model,condition=condition,side='candidate',
            completed=0,total=300,attempt=str(out),production_started=False,native_gate_passed=False)
        def update():
            state['updated_at']=time.time();write(out/'status.json',state)
            write(folder/'telemetry/natural_candidate_state.json',state)
        update()
        first={}
        for job in jobs:
            if job['repeat']==0:first.setdefault(job['task'],job['job_key'])
        ordered=sorted(jobs,key=lambda j:0 if j['job_key'] in first.values() else 1)
        processed=0
        for job in ordered:
            if integration_smoke and job['job_key'] not in first.values():break
            directory=out/job['job_key'];directory.mkdir(exist_ok=True)
            result_path=directory/'result.json'
            job_binding=dict(job=job,input_sha256=canonical(rows[job['uid']]),identity=identity)
            if result_path.exists():
                r=read(result_path)
                if r.get('binding_sha256')!=canonical(job_binding) or not r.get('record_complete'):raise Blocked('Existing success changed')
                done+=1;continue
            if list(directory.glob('failure-*.json')):raise Blocked('Failed calibration job requires explicit recovery')
            try:
                if session is None:session=NativeSession(folder,model,condition,out/'loading.json')
                state.update(job_key=job['job_key'],completed=done);update()
                generated=session.generate(rows[job['uid']],job,directory/'prefix.json')
                count=len(generated['generated_token_ids']);cap=job['generation']['max_new_tokens']
                if count!=generated['output_tokens'] or not 0<count<=cap:raise Blocked('Invalid output count')
                if generated['output_cap_reached']!=(count==cap) or (not generated['eos_terminated'] and count!=cap):raise Blocked('Invalid termination')
                result=dict(job);result.update(generated,record_complete=True,binding_sha256=canonical(job_binding),
                    completed_at=time.time(),execution_identity=identity)
                publish_once(result_path,result);done+=1;processed+=1
            except BaseException as exc:
                publish_once(directory/('failure-'+str(time.time_ns())+'.json'),dict(error_type=type(exc).__name__,error=str(exc)[:1500],binding=job_binding))
                state.update(stage='failed',exit_code=1,completed=done);update();raise
        state.update(stage='integration_smoke_complete' if integration_smoke else 'complete',exit_code=0,
                     completed=done,wall_seconds=time.time()-started,full_matrix_complete=not integration_smoke);update()
        publish_once(out/('integration_smoke.json' if integration_smoke else 'completion.json'),state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--model',required=True)
    p.add_argument('--condition',required=True);p.add_argument('--out',required=True);p.add_argument('--integration-smoke',action='store_true');a=p.parse_args()
    run(a.folder,a.model,a.condition,a.out,a.integration_smoke)
