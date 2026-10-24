"""Prepare existing pools then audit both CPU input paths without waiting for GQA."""
import argparse,fcntl,os,subprocess,time
from pathlib import Path
from .common import write


def run(folder):
    lock=(folder/'existing-audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=folder/'tests'/('existing-inputs-'+str(time.time_ns()));attempt.mkdir()
    state=dict(pid=os.getpid(),production_started=False,stage='prepare_candidates',attempt=str(attempt))
    def update(**kw):
        state.update(kw,updated_at=time.time());write(folder/'telemetry/existing_audit_state.json',state)
    children=[]
    try:
        update()
        subprocess.run(['/root/autodl-tmp/Compress/.venv/bin/python','-m','r3.existing_candidates','--folder',str(folder),'--base','/root/autodl-tmp/Compress/data'],check=True)
        source=folder/'data/existing-pools-v1/candidates_unvalidated.jsonl'
        for model in ['qwen','llava']:
            child=subprocess.Popen([str(folder/('environment/native-'+model+'/bin/python')),'-m','r3.audit_'+model+'_inputs','--folder',str(folder),'--candidates',str(source),'--out',str(attempt/(model+'.jsonl'))])
            children.append(child)
        update(stage='auditing_inputs',child_pids=[p.pid for p in children])
        codes=[p.wait() for p in children]
        if any(codes):raise RuntimeError('Input audit exit codes '+str(codes))
        update(stage='complete_qwen_llava_mini_pending',exit_code=0)
    except BaseException as exc:
        update(stage='failed',exit_code=1,error_type=type(exc).__name__);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();run(Path(a.folder))
