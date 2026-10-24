"""Five-task A integration only; never repeats previously completed long tests."""
import argparse
import os
import subprocess
import time
from pathlib import Path
from .common import Blocked,read,write


def run(root):
    import fcntl
    root=Path(root)
    with (root/'llava_integration_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=read(root/'telemetry/natural_integration_launch.json')
        out=root/'tests'/('llava-integration-'+str(time.time_ns()));out.mkdir()
        state=dict(pid=os.getpid(),attempt=str(out),stage='waiting_existing_queue',results=[],production_started=False,native_gate_passed=False)
        def update():state['updated_at']=time.time();write(root/'telemetry/llava_integration_state.json',state)
        update()
        while Path('/proc/'+str(prior['pid'])).exists():time.sleep(10)
        # Independent A checks can proceed even if the preceding offline parser
        # failed; record that failure, without hiding or retrying it.
        state['previous_queue_state']=read(root/'telemetry/natural_integration_state.json')['stage']
        def execute(name,env,module,args):
            with (out/(name+'.log')).open('xb') as log:
                p=subprocess.Popen([str(root/('environment/native-'+env+'/bin/python')),'-u','-m',module]+args,stdout=log,stderr=log)
                state.update(stage='running',current=name,child_pid=p.pid);update();rc=p.wait()
            state['results'].append(dict(name=name,exit_code=rc));update();return rc
        for condition in ('Full','FastV25','PriorTR25','AgilePruner25','VisionZip25'):
            env='fastv' if condition=='FastV25' else 'llava'
            candidate=out/(condition+'-candidate');reference=out/(condition+'-reference')
            rc=execute(condition+'-candidate',env,'r3.natural_candidate',
                ['--folder',str(root),'--model','llava7','--condition',condition,'--out',str(candidate),'--integration-smoke'])
            if rc:continue
            if condition=='FastV25':
                module='r3.fastv_natural_reference';args=['--folder',str(root),'--condition',condition,'--out',str(reference),'--integration-smoke']
            else:
                module='r3.llava_natural_reference';args=['--folder',str(root),'--manifest',str(root/'manifests/frozen-inputs-v1/calibration_manifest.jsonl'),
                    '--condition',condition,'--out',str(reference),'--integration-smoke']
            if execute(condition+'-reference',env,module,args):continue
            analysis=out/(condition+'-analysis')
            rc=execute(condition+'-analysis',env,'r3.natural_smoke_analysis',
                ['--folder',str(root),'--model','llava7','--condition',condition,'--attempts',str(candidate),str(reference),'--out',str(analysis)])
            if not rc and not read(analysis/'completion.json').get('offline_parsing_and_scoring_passed'):
                state['results'].append(dict(name=condition+'-analysis-receipt',exit_code=1));update()
        state.update(stage='complete',exit_code=int(any(r['exit_code'] for r in state['results'])));update();write(out/'completion.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);run(p.parse_args().folder)
