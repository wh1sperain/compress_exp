"""Serial diagnostic jobs on one GPU, isolated receipts; never launches research."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from .common import read,write,digest


def run(folder,selectors_only=False,methods=None,resume_only=False,retry_failed_attempt=None):
    lock=(folder/'correctness.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=folder/'tests'/(('model-resume-' if resume_only else 'real-selectors-' if selectors_only else 'core-correctness-')+str(time.time_ns()));attempt.mkdir()
    candidates=[json.loads(s) for s in (folder/'data/existing-pools-v1/candidates_unvalidated.jsonl').read_text().splitlines()]
    row=min((r for r in candidates if r['task']=='DocVQA'),key=lambda r:digest(r['uid']))
    # Only image/question/options/provenance enter the diagnostic worker.
    candidate=attempt/'candidate.json';write(candidate,{k:row[k] for k in ['uid','question','image_path','image_sha256','options'] if k in row})
    jobs=[('fastv','FastVFull'),('fastv','FastV50'),('fastv','FastV25'),
          ('llava','Full'),('llava','AgileFull'),('llava','VisionZipFull'),
          ('llava','PriorTR50'),('llava','PriorTR25'),('llava','Agile50'),('llava','Agile25'),
          ('llava','VisionZip50'),('llava','VisionZip25'),('mini','MiniNative'),
          ('qwen','Full'),('qwen','PriorTR50'),('qwen','PriorTR25')]
    if selectors_only:jobs=[(e,m) for e,m in jobs if m.startswith(('PriorTR','Agile','VisionZip')) and not m.endswith('Full')]
    if methods:
        if not selectors_only:raise ValueError('Method filtering restricted to selector diagnostics')
        if not set(methods)<={m for e,m in jobs}:raise ValueError('Unregistered diagnostic methods')
        jobs=[(e,m) for e,m in jobs if m in methods]
    if retry_failed_attempt:
        if not resume_only:raise ValueError('Recovery filter is diagnostic only')
        prior=read(Path(retry_failed_attempt)/'completion.json')
        failed={r['name'] for r in prior['results'] if r['exit_code']!=0}
        if not failed:raise ValueError('No failed diagnostics to recover')
        jobs=[(e,m) for e,m in jobs if e+'-'+m in failed]
    state={'retry_failed_attempt':retry_failed_attempt,'pid':os.getpid(),'attempt':str(attempt),'production_started':False,'results':[]}
    def update(**kw):state.update(kw,updated_at=time.time());write(folder/('telemetry/model_resume_state.json' if resume_only else 'telemetry/real_selectors_state.json' if selectors_only else 'telemetry/correctness_state.json'),state)
    update(stage='running')
    for env,method in jobs:
        name=env+'-'+method;out=attempt/name
        module='r3.qwen_smoke' if env=='qwen' else 'r3.native_smoke'
        command=[str(folder/('environment/native-'+env+'/bin/python')),'-u','-m',module,
                 '--folder',str(folder),'--method',method,'--out',str(out),'--candidate',str(candidate)]
        if selectors_only:command.append('--selector-only')
        if resume_only:command.append('--resume-only')
        with (attempt/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            update(current=name,child_pid=child.pid)
            rc=child.wait()
        state['results'].append({'name':name,'exit_code':rc,'status':'FAIL' if rc else 'INCONCLUSIVE'})
        update()
    update(stage='diagnostics_complete',exit_code=0,native_gate_passed=False)
    write(attempt/'completion.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--selectors-only',action='store_true')
    p.add_argument('--methods',nargs='+')
    p.add_argument('--resume-only',action='store_true')
    p.add_argument('--retry-failed-attempt')
    a=p.parse_args();run(Path(a.folder),a.selectors_only,a.methods,a.resume_only,a.retry_failed_attempt)
