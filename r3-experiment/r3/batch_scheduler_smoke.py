"""Five fixed calibration requests, original caps, real EOS and slot reuse.

Raw execution smoke only, not formal data, answer analysis or equivalence proof.
"""
import argparse,json,time
from pathlib import Path
from .common import read,write,digest,code_hash,seed,Blocked
from .batch_backend_probe import choose_rows
from .native_session import NativeSession
from .batch_scheduler import Scheduler
from .runner import publish_once

def run(folder,model,condition,out):
    folder=Path(folder);out=Path(out)
    if out.exists():raise Blocked('Existing smoke evidence is immutable')
    out.mkdir(parents=True)
    spec=read(Path(__file__).resolve().parent.parent/'round3_spec.json')
    rows=[json.loads(s) for s in (folder/'manifests/frozen-inputs-v1/calibration_manifest.jsonl').read_text().splitlines()]
    selected=choose_rows(rows,5)
    session=NativeSession(folder,model,condition,out/'loading.json')
    profile=dict(backend='native-padded-batch',dtype='bfloat16',kv_dtype='bfloat16',max_live_requests=4,kv_scratch_factor=3)
    scheduler=Scheduler(session,profile);requests=[]
    for row in selected:
        key=digest('scheduler-smoke|'+model+'|'+condition+'|'+row['uid'])
        job=dict(model=model,condition=condition,uid=row['uid'],job_key=key,seed=seed(row['uid'],0),
                 generation=dict(spec['generation'],max_new_tokens=spec['models'][model]['max_new_tokens']))
        requests.append((row,job,out/key/'prefix.json'))
    started=time.time();completed=[]
    try:
        for job,row,generated in scheduler.run(requests,out/'batch_status.json'):
            from .batch_runner import validate_generation
            validate_generation(generated,job)
            publish_once(out/job['job_key']/'result.json',dict(job,**generated,diagnostic_only=True,labels_read=False))
            completed.append(dict(uid=row['uid'],tokens=generated['output_tokens'],eos=generated['eos_terminated'],cap=generated['output_cap_reached']))
        status=read(out/'batch_status.json')
        if len(completed)!=5 or status['peak_live']<2:raise Blocked('Five requests and actual multi-slot operation required')
        receipt=dict(status='PASS',model=model,condition=condition,completed=completed,peak_live=status['peak_live'],
                     wall_seconds=time.time()-started,scope='original-cap calibration raw execution and slot reuse only',
                     full_native_engine_certification=False,code_sha256=code_hash(),parsing_enabled=False,scoring_enabled=False,
                     backend_impl_sha256={'r3/'+p.name:digest(p.read_bytes()) for p in [Path(__file__).parent/n for n in
                     ['llava_batch_decode.py','qwen_batch_decode.py','mini_batch_decode.py','batch_scheduler.py']]})
        write(out/'receipt.json',receipt);return receipt
    except BaseException as exc:
        write(out/'failure.json',dict(status='FAIL',type=type(exc).__name__,error=str(exc)[:1200],completed=completed));raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('folder','model','condition','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(run(a.folder,a.model,a.condition,a.out)))
