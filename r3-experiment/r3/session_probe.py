"""Validate persistent worker inputs on calibration images; not research or a gate."""
import argparse
import json
import time
from pathlib import Path
from .common import read,write,digest,Blocked,code_hash
from .manifest import TASKS


def run(folder,model,condition,out):
    from .native_session import NativeSession
    out=Path(out)
    if out.exists():raise Blocked('Existing diagnostic output')
    out.mkdir(parents=True)
    manifest=Path(folder)/'manifests/frozen-inputs-v1/calibration_manifest.jsonl'
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    selected=[min((r for r in rows if r['task']==t),key=lambda r:digest(r['uid'])) for t in TASKS]
    session=NativeSession(folder,model,condition,out/'loading.json')
    records=[]
    try:
        for row in selected:
            job={'job_key':'diagnostic-'+model+'-'+condition+'-'+row['uid'],'seed':1234,
                 'generation':dict(max_new_tokens=1,temperature=1.,top_k=20,top_p=.95)}
            result=session.generate(row,job,out/(digest(row['uid'])+'.prefix.json'))
            records.append(dict(uid=row['uid'],task=row['task'],**result))
            write(out/'status.json',{'stage':'running','completed':len(records),'updated_at':time.time()})
        write(out/'report.json',{'status':'PASS','scope':'five calibration image input hashes and one-step runtime path',
              'native_gate_passed':False,'code_sha256':code_hash(),'manifest_sha256':digest(manifest.read_bytes()),'records':records})
        write(out/'status.json',{'stage':'complete','completed':len(records),'native_gate_passed':False,'updated_at':time.time()})
    except BaseException as exc:
        write(out/'status.json',{'stage':'failed','completed':len(records),'error':str(exc)[:1200]});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--model',required=True)
    p.add_argument('--condition',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    run(a.folder,a.model,a.condition,a.out)
