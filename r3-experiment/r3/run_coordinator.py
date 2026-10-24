"""Sequential isolated workers, preserving the registered complete job set."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from .common import read,write,digest,code_hash,Blocked
from .runner import publish_once,load_plan


def run(folder,registration,out):
    folder=Path(folder);out=Path(out);registration=Path(registration)
    with (folder/'production-coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r=read(registration)
        from .execution_acceptance import MODE
        reduced=r.get('execution_mode')==MODE
        if r.get('code_sha256')!=code_hash():raise Blocked('Stale registration')
        if reduced:
            if r.get('gates_verified') is not False or r.get('full_native_engine_certification') is not False or r.get('inference_only') is not True:
                raise Blocked('Reduced registration must disclose missing full certification')
            if r['inventory']['planned_records']!=21900:raise Blocked('Reduced registration must preserve full main+controls scope')
        elif not r.get('gates_verified'):raise Blocked('Unverified strict registration')
        seen=set();total=0
        for item in r['plans']:
            path=Path(item['path'])
            if digest(path.read_bytes())!=item['sha256']:raise Blocked('Registered plan changed')
            plan,_,_=load_plan(path,folder)
            if reduced and (plan.get('execution_mode')!=MODE or plan.get('execution_acceptance_sha256')!=r.get('execution_acceptance_sha256')):
                raise Blocked('Plan/registration execution acceptance mismatch')
            if not reduced and plan.get('execution_mode'):raise Blocked('Mixed registration execution policies')
            if len(plan['jobs'])!=item['records']:raise Blocked('Plan count mismatch')
            for job in plan['jobs']:
                if job['job_key'] in seen:raise Blocked('Job repeated across plans')
                seen.add(job['job_key'])
            total+=len(plan['jobs'])
        if total!=r['inventory']['planned_records']:raise Blocked('Incomplete registration')
        out.mkdir(parents=True,exist_ok=True)
        binding={'registration_sha256':digest(registration.read_bytes()),'code_sha256':code_hash()}
        path=out/'binding.json'
        if path.exists():
            if read(path)!=binding:raise Blocked('Coordinator binding changed')
        else:publish_once(path,binding)
        state={'pid':os.getpid(),'stage':'running','results':[],'planned_records':total}
        if reduced:state.update(execution_mode=MODE,full_native_engine_certification=False,inference_only=True,scoring_status='disabled_by_user')
        def update():state['updated_at']=time.time();write(out/'status.json',state)
        update()
        for item in r['plans']:
            name=Path(item['path']).stem
            cmd=[str(folder/('environment/native-'+item['environment']+'/bin/python')),
                 '-u','-m','r3.runner','--folder',str(folder),'--plan',item['path'],'--out',str(out/name)]
            log=out/(name+'-'+str(time.time_ns())+'.log')
            with log.open('xb') as f:
                child=subprocess.Popen(cmd,stdout=f,stderr=f)
                state.update(current=name,child_pid=child.pid);update();rc=child.wait()
            state['results'].append({'plan':name,'exit_code':rc});update()
            if rc:
                state.update(stage='failed',exit_code=rc);update()
                raise Blocked('Worker failed; no automatic answer resampling')
        state.update(stage='generation_complete',exit_code=0,analysis_complete=False);update()
        publish_once(out/'completion.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--registration',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.folder,a.registration,a.out)
