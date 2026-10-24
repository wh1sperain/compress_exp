"""Isolated persistent-adapter checks, with a shared exclusive diagnostic lock."""
import argparse,fcntl,os,subprocess,time
from pathlib import Path
from .common import write


def run(folder):
    with (folder/'correctness.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=folder/'tests'/('worker-session-'+str(time.time_ns()));attempt.mkdir()
        jobs=[('fastv','llava7',m) for m in ('FastV50','FastV25')]
        jobs += [('llava','llava7',m) for m in ('Full','PriorTR50','PriorTR25','Agile50','Agile25','VisionZip50','VisionZip25')]
        jobs += [('qwen','qwen8',m) for m in ('Full','PriorTR50','PriorTR25')]+[('mini','mini8','MiniNative')]
        state={'pid':os.getpid(),'attempt':str(attempt),'stage':'running','results':[],'production_started':False}
        def update():state['updated_at']=time.time();write(folder/'telemetry/worker_session_state.json',state)
        update()
        for env,model,condition in jobs:
            name=model+'-'+condition
            cmd=[str(folder/('environment/native-'+env+'/bin/python')),'-u','-m','r3.session_probe',
                 '--folder',str(folder),'--model',model,'--condition',condition,'--out',str(attempt/name)]
            with (attempt/(name+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=log)
                state.update(current=name,child_pid=child.pid);update();rc=child.wait()
            state['results'].append({'name':name,'exit_code':rc});update()
        state.update(stage='complete',exit_code=int(any(r['exit_code'] for r in state['results'])),native_gate_passed=False)
        update();write(attempt/'completion.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();run(Path(a.folder))
