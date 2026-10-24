"""Explicit diagnostic recovery; preserve original failure and session evidence."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from .common import read, write, Blocked
from .parallel_verification import execution_result


def run(root):
    import fcntl
    with (root/'correctness.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=read(root/'telemetry/integrated_verification_state.json')
        if prior['stage']!='complete' or prior['exit_code']!=1:
            raise Blocked('Expected preserved failed diagnostic')
        old=Path(prior['attempt']); session=old/'mini-session'
        if not (old/'mini-author.failure.json').exists() or not (session/'receipt.json').exists():
            raise Blocked('Missing original failure or reusable session evidence')
        attempt=root/'tests'/('mini-author-recovery-'+str(time.time_ns()));attempt.mkdir()
        state=dict(pid=os.getpid(),attempt=str(attempt),original_attempt=str(old),
                   stage='running',production_started=False,native_gate_passed=False,results=[])
        def update():
            state['updated_at']=time.time();write(root/'telemetry/mini_recovery_state.json',state)
        exe=str(root/'environment/native-mini/bin/python')
        author=attempt/'author';comparison=attempt/'comparison.json'
        uid=read(session/'receipt.json')['uid']
        commands=[('author',['--mode','author','--folder',str(root),'--manifest',str(root/'manifests/frozen-inputs-v1/calibration_manifest.jsonl'),
                            '--uid',uid,'--out',str(author)],None),
                  ('compare',['--mode','compare','--left',str(author),'--right',str(session),'--out',str(comparison)],comparison)]
        for name,args,receipt in commands:
            with (attempt/(name+'.log')).open('xb') as log:
                child=subprocess.Popen([exe,'-u','-m','r3.mini_independent_probe']+args,stdout=log,stderr=log)
                state.update(current=name,child_pid=child.pid);update();rc=child.wait()
            result=execution_result(name,rc,receipt);state['results'].append(result);update()
            if result['exit_code']:break
        state.update(stage='complete',exit_code=int(any(r['exit_code'] for r in state['results'])))
        update();write(attempt/'completion.json',state)
        return state['exit_code']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True)
    raise SystemExit(run(p.parse_args().folder))
