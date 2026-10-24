"""Integrate parallel-developed checks on one GPU, isolated native processes."""
import argparse,json,os,subprocess,time
from pathlib import Path
from .common import read,write,digest,Blocked


def execution_result(name, process_exit_code, comparison_receipt=None):
    """Exit zero means the comparator ran, not that its comparison passed.

    Inspect but never rewrite the original receipt. Missing, malformed, FAIL,
    and INCONCLUSIVE receipts all fail closed; preserve the subprocess code.
    """
    result={'name':name,'process_exit_code':process_exit_code,
            'exit_code':process_exit_code,'status':'PASS' if process_exit_code==0 else 'FAIL'}
    if comparison_receipt is None:return result
    path=Path(comparison_receipt)
    result['comparison_receipt']=str(path)
    try:
        raw=path.read_bytes()
        result['comparison_receipt_sha256']=digest(raw)
        receipt=json.loads(raw)
        if not isinstance(receipt,dict):raise ValueError('Comparison receipt must be an object')
        result['comparison_status']=receipt.get('status')
        if receipt.get('status')!='PASS':
            result.update(status='FAIL',exit_code=process_exit_code or 1,
                          reason='Comparison receipt did not report PASS')
    except (OSError,ValueError,UnicodeError) as exc:
        result.update(status='FAIL',exit_code=process_exit_code or 1,
                      reason='Comparison receipt unreadable or malformed',
                      receipt_error=type(exc).__name__)
    return result


def run(folder):
    import fcntl  # Linux server lock; receipt validation remains CPU-testable on Windows.
    with (folder/'correctness.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=folder/'tests'/('integrated-verification-'+str(time.time_ns()));attempt.mkdir()
        manifest=folder/'manifests/frozen-inputs-v1/calibration_manifest.jsonl'
        rows=[json.loads(s) for s in manifest.read_text().splitlines()]
        row=min((r for r in rows if r['task']=='DocVQA'),key=lambda r:digest(r['uid']))
        candidate=attempt/'candidate.json'
        write(candidate,{k:v for k,v in row.items() if k in ('uid','question','options','image_path','image_sha256')})
        state={'pid':os.getpid(),'attempt':str(attempt),'stage':'running','results':[],'production_started':False}
        def update():state['updated_at']=time.time();write(folder/'telemetry/integrated_verification_state.json',state)
        def execute(name,env,module,args,comparison_receipt=None):
            cmd=[str(folder/('environment/native-'+env+'/bin/python')),'-u','-m',module]+args
            with (attempt/(name+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=log)
                state.update(current=name,child_pid=child.pid);update();rc=child.wait()
            result=execution_result(name,rc,comparison_receipt)
            state['results'].append(result);update();return result['exit_code']
        update()
        for condition in ('FastV75','FastV37p5','Random25_L2'):
            execute(condition,'fastv','r3.session_probe',['--folder',str(folder),'--model','llava7',
                    '--condition',condition,'--out',str(attempt/condition)])
        supplemental=read(folder/'telemetry/qwen4_supplemental_audit_launch.json')
        for condition in ('Full','PriorTR25'):
            execute('qwen4-'+condition,'qwen','r3.session_probe',['--folder',str(folder),'--model','qwen4',
                '--condition',condition,'--out',str(attempt/('qwen4-'+condition)),
                '--supplemental-audit',supplemental['out'],'--supplemental-status',supplemental['status_path']])
        for method in ('Full','PriorTR50','PriorTR25'):
            paths={side:attempt/('qwen-'+method+'-'+side) for side in ('author','local')}
            codes=[]
            for side,path in paths.items():
                codes.append(execute(path.name,'qwen','r3.qwen_independent_probe',
                    ['--folder',str(folder),'--candidate',str(candidate),'--method',method,'--side',side,'--out',str(path)]))
            if not any(codes):
                execute('qwen-'+method+'-compare','qwen','r3.qwen_independent_probe',
                    ['--compare',str(paths['author']),str(paths['local']),'--out',str(attempt/('qwen-'+method+'-comparison.json'))],
                    comparison_receipt=attempt/('qwen-'+method+'-comparison.json'))
        paths={side:attempt/('mini-'+side) for side in ('author','session')};codes=[]
        for side,path in paths.items():
            codes.append(execute(path.name,'mini','r3.mini_independent_probe',['--mode',side,'--folder',str(folder),
                '--manifest',str(manifest),'--uid',row['uid'],'--out',str(path)]))
        if not any(codes):
            execute('mini-compare','mini','r3.mini_independent_probe',['--mode','compare','--left',str(paths['author']),
                '--right',str(paths['session']),'--out',str(attempt/'mini-comparison.json')],
                comparison_receipt=attempt/'mini-comparison.json')
        state.update(stage='complete',exit_code=int(any(r['exit_code'] for r in state['results'])),native_gate_passed=False)
        update();write(attempt/'completion.json',state)
        return state['exit_code']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();raise SystemExit(run(Path(a.folder)))
