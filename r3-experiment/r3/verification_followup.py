"""Recorded recovery of FastV worker checks followed by official-entry comparison."""
import argparse,fcntl,os,subprocess,time
from pathlib import Path
from .common import read,write,Blocked


def run(folder):
    with (folder/'correctness.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=read(folder/'telemetry/worker_session_state.json')
        if prior['stage']!='complete':raise Blocked('Prior worker checks still active')
        failures={r['name'] for r in prior['results'] if r['exit_code']}
        if failures!={'llava7-FastV50','llava7-FastV25'}:raise Blocked('Unexpected failures; inspect before recovery')
        attempt=folder/'tests'/('official-followup-'+str(time.time_ns()));attempt.mkdir()
        state={'pid':os.getpid(),'stage':'running','attempt':str(attempt),'results':[],
               'original_worker_attempt':prior['attempt'],'production_started':False}
        def update():state['updated_at']=time.time();write(folder/'telemetry/official_followup_state.json',state)
        update()
        jobs=[(method,'r3.session_probe',['--model','llava7','--condition',method]) for method in ('FastV50','FastV25')]
        jobs += [('official-entry','r3.fastv_official_probe',[])]
        for name,module,extra in jobs:
            cmd=[str(folder/'environment/native-fastv/bin/python'),'-u','-m',module,
                 '--folder',str(folder),'--out',str(attempt/name)]+extra
            with (attempt/(name+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=log)
                state.update(current=name,child_pid=child.pid);update();rc=child.wait()
            state['results'].append({'name':name,'exit_code':rc});update()
            if rc:break
        state.update(stage='complete',exit_code=int(any(r['exit_code'] for r in state['results'])),native_gate_passed=False)
        update();write(attempt/'completion.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();run(Path(a.folder))
