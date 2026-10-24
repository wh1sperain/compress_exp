"""CPU-only aggregation and offline checks after completed natural smoke attempts.

Does not poll, start, stop, or otherwise manage GPU workers. Partial smoke
records are compared against the full 600-row condition plan, never certified
as a complete natural comparison.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from .common import Blocked,digest,read
from .enrich_records import file_digest
from .natural_preparation import build
from .offline_pipeline import frozen_inputs,run_pipeline
from .runner import publish_once


def collect_attempts(attempts,model,condition,expected):
    files={};receipts=[];counts=Counter()
    for directory in map(Path,attempts):
        candidates=[directory/'integration_smoke.json',directory/'completion.json']
        available=[p for p in candidates if p.exists()]
        if not available:raise Blocked('Attempt has no terminal receipt: '+str(directory))
        terminal=read(available[-1])
        if terminal.get('exit_code',0)!=0:raise Blocked('Attempt exited unsuccessfully')
        if terminal.get('stage') in ('failed','running'):raise Blocked('Attempt is not complete')
        if list(directory.glob('*/failure-*.json')):raise Blocked('Attempt contains preserved failed jobs; do not conceal them in smoke analysis')
        paths=sorted(directory.glob('*/result.json'))
        if not paths or terminal.get('completed')!=len(paths):raise Blocked('Terminal receipt/actual record count mismatch')
        for path in paths:
            record=read(path);key=record['job_key']
            if key in files:raise Blocked('Duplicate original job across attempts; do not choose one answer')
            if record.get('model')!=model or record.get('condition')!=condition or key not in expected:
                raise Blocked('Attempt record outside requested natural condition')
            if not record.get('record_complete') or record.get('stage')!='natural_certification':
                raise Blocked('Not a complete calibration record')
            for name,value in expected[key].items():
                if name=='record_complete':continue
                if record.get(name)!=value:raise Blocked('Record differs from frozen natural plan: '+name)
            files[key]=path;counts[record['side']]+=1
        receipts.append({'attempt':str(directory),'terminal_receipt':str(available[-1]),
                         'terminal_sha256':file_digest(available[-1]),'records':len(paths)})
    return files,receipts,dict(counts)


def run(folder,attempts,out,model,condition,tokenizer,tokenizer_provenance):
    folder,out=Path(folder),Path(out)
    if out.exists():raise Blocked('Analysis directory immutable; choose a fresh attempt')
    frozen=folder/'manifests/frozen-inputs-v1'
    calibration,_,locked=frozen_inputs(frozen,'calibration')
    spec=read(Path(__file__).resolve().parent.parent/'round3_spec.json')
    matrix=build(spec,calibration,locked['manifest_sha256'],include_qwen4=model=='qwen4')
    expected={j['job_key']:j for j in matrix['jobs'] if j['model']==model and j['condition']==condition}
    if len(expected)!=600:raise Blocked('Expected one full 600-row natural comparison')
    files,receipts,counts=collect_attempts(attempts,model,condition,expected)
    out.mkdir(parents=True)
    plan_path=out/'natural_preparation.json';publish_once(plan_path,matrix)
    raw_path=out/'original_records.jsonl'
    raw_sources=[]
    with raw_path.open('x',encoding='utf-8') as output:
        for key,path in sorted(files.items()):
            record=read(path)
            output.write(json.dumps(record,ensure_ascii=False)+'\n')
            raw_sources.append({'job_key':key,'path':str(path),'sha256':file_digest(path)})
    publish_once(out/'aggregation.json',{'sources':raw_sources,'attempt_receipts':receipts,'side_counts':counts,
                 'actual_records':len(files),'full_expected_records':600,'original_records_sha256':file_digest(raw_path)})
    report=run_pipeline(raw_path,frozen,'calibration',plan_path,out/'offline',model,tokenizer,tokenizer_provenance,condition)
    result={'status':'INCONCLUSIVE' if len(files)<600 else 'FULL_MATRIX_ANALYZED',
            'model':model,'condition':condition,'records':len(files),'expected':600,'side_counts':counts,
            'offline_parsing_and_scoring_passed':not report['failures'] and report['scored_records']==len(files),
            'records_complete':report['records_complete'],'natural_certification':report['natural_certification'],
            'native_gate_passed':False,'production_ready':False,
            'scope':'Completed smoke record pipeline check only; partial records cannot certify natural behavior.'}
    publish_once(out/'completion.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--attempts',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=['llava7','mini8','qwen8','qwen4'],required=True)
    p.add_argument('--condition',required=True);a=p.parse_args()
    lock=read(a.folder/'manifests/model_weights_lock.json')[a.model];path=Path(lock['path'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(path),use_fast=a.model.startswith('qwen'),local_files_only=True,trust_remote_code=False)
    names=('tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt')
    provenance={'path':str(path),'revision':lock['revision'],'class':type(tokenizer).__name__,
                'files':{name:file_digest(path/name) for name in names if (path/name).is_file()}}
    print(json.dumps(run(a.folder,a.attempts,a.out,a.model,a.condition,tokenizer,provenance)))


if __name__=='__main__':main()
