"""Finish distinct-path integration checks in serial GPU / offline CPU stages."""
import argparse
import os
import subprocess
import time
from pathlib import Path
from .common import Blocked, read, write


def run(root):
    import fcntl
    root=Path(root)
    with (root/'natural_integration_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        launch=read(root/'telemetry/natural_candidate_launch.json')
        candidate=Path(launch['attempt'])
        out=root/'tests'/('natural-integration-'+str(time.time_ns()));out.mkdir()
        state=dict(pid=os.getpid(),stage='waiting_existing_candidate',attempt=str(out),
            reused_candidate=str(candidate),results=[],production_started=False,native_gate_passed=False)
        def update():
            state['updated_at']=time.time();write(root/'telemetry/natural_integration_state.json',state)
        update()
        try:
            while Path('/proc/'+str(launch['pid'])).exists():
                time.sleep(10)
            smoke=read(candidate/'integration_smoke.json')
            if smoke.get('exit_code')!=0 or smoke.get('completed')!=5:raise Blocked('Candidate integration did not finish five tasks')
            def execute(name,env,module,args):
                with (out/(name+'.log')).open('xb') as log:
                    child=subprocess.Popen([str(root/('environment/native-'+env+'/bin/python')),'-u','-m',module]+args,stdout=log,stderr=log)
                    state.update(stage='running',current=name,child_pid=child.pid);update();rc=child.wait()
                state['results'].append(dict(name=name,exit_code=rc));update()
                if rc:raise Blocked('Stage failed: '+name)
                if name.endswith('offline-analysis'):
                    receipt=read(Path(args[args.index('--out')+1])/'completion.json')
                    if receipt.get('offline_parsing_and_scoring_passed') is not True:
                        raise Blocked('Offline parsing/scoring failed: '+name)
            reference=out/'qwen-full-reference'
            execute('qwen-full-reference','qwen','r3.qwen_natural_reference',
                ['--folder',str(root),'--model','qwen8','--condition','Full','--out',str(reference),'--integration-smoke'])
            # No new model inference: score the completed paired five-task rows.
            execute('qwen-offline-analysis','qwen','r3.natural_smoke_analysis',
                ['--folder',str(root),'--model','qwen8','--condition','Full','--attempts',str(candidate),str(reference),'--out',str(out/'qwen-analysis')])
            mini=out/'mini-reference'
            proof=Path(read(root/'telemetry/mini_recovery_state.json')['attempt'])/'comparison.json'
            execute('mini-reference','mini','r3.mini_natural_reference',
                ['--folder',str(root),'--manifest',str(root/'manifests/frozen-inputs-v1/calibration_manifest.jsonl'),
                 '--numeric-receipt',str(proof),'--out',str(mini),'--smoke'])
            execute('mini-offline-analysis','mini','r3.natural_smoke_analysis',
                ['--folder',str(root),'--model','mini8','--condition','MiniNative','--attempts',str(mini),'--out',str(out/'mini-analysis')])
            state.update(stage='complete',exit_code=0,full_natural_certification=False);update();write(out/'completion.json',state)
        except BaseException as exc:
            state.update(stage='failed',exit_code=1,error=str(exc)[:1800]);update();write(out/'failure.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);run(p.parse_args().folder)
