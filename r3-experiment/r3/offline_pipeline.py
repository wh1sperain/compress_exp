"""Offline immutable records -> original-token regions -> scoring -> certification.

Reads isolated labels only here. Missing/failed rows never become zero scores.
Natural statistics require the full expected reference/candidate matrix.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from .common import Blocked,digest,read
from .enrich_records import enrich_record,file_digest
from .runner import publish_once
from .task_scoring import score_record,labels_as_list


def frozen_inputs(folder,split):
    folder=Path(folder);name={'main':'main_manifest.jsonl','calibration':'calibration_manifest.jsonl'}[split]
    lock=read(folder/'manifest_lock.json');manifest=folder/name;label_path=folder/'scoring_labels.json'
    for path in (manifest,label_path):
        if lock['hashes'].get(path.name)!=file_digest(path):raise Blocked('Frozen input hash mismatch: '+path.name)
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    index={row['uid']:row for row in rows};expected=500 if split=='main' else 100
    if len(rows)!=expected or len(index)!=expected:raise Blocked('Frozen manifest count mismatch')
    labels=read(label_path)
    if not isinstance(labels,dict):raise Blocked('Frozen labels must be UID -> answer dictionary')
    for uid in index:
        if uid not in labels or not isinstance(labels[uid],dict):raise Blocked('Missing isolated reference')
        labels_as_list(labels[uid])
    return rows,labels,{'manifest_sha256':file_digest(manifest),'labels_sha256':file_digest(label_path),
                       'manifest_lock_sha256':file_digest(folder/'manifest_lock.json')}


def process_record(record,row,label,job,tokenizer):
    for field in ('job_key','model','uid','condition','repeat','seed','task','image_sha256','generation'):
        if record.get(field)!=job.get(field):raise Blocked('Output/registered job mismatch: '+field)
    for field in ('side','stage'):
        if field in job and record.get(field)!=job[field]:raise Blocked('Output job scope mismatch: '+field)
    for field in ('uid','task','image_sha256'):
        if row[field]!=record[field]:raise Blocked('Output/frozen input mismatch: '+field)
    if 'prompt_ids' not in record or 'generation_eos_token_id' not in record:
        raise Blocked('Need actual prompt IDs and generation EOS receipt')
    audit=row.get('native_inputs',{}).get(record['model'])
    if audit and audit.get('prompt_sha256')!=record['prompt_sha256']:
        raise Blocked('Output prompt differs from frozen native audit')
    enriched=enrich_record(record,{'model':record['model'],'uid':record['uid'],
                                  'prompt_ids':record['prompt_ids']},tokenizer,record['generation_eos_token_id'])
    score=score_record(enriched,row,label)
    if score['status']!='scored':raise Blocked('Scoring did not complete')
    merged=dict(enriched)
    for field,value in score.items():
        if field in merged and merged[field]!=value:raise Blocked('Scoring would replace original field: '+field)
        merged[field]=value
    merged['think_ended']=enriched['thinking_boundary_found']
    return merged


def run_pipeline(records_path,frozen,split,jobs_path,out,model,tokenizer,tokenizer_provenance,condition=None):
    records_path,jobs_path,out=map(Path,(records_path,jobs_path,out))
    if out.exists():raise Blocked('Offline pipeline output directory immutable')
    manifest,labels,provenance=frozen_inputs(frozen,split);by_uid={r['uid']:r for r in manifest}
    plan=read(jobs_path);all_jobs=plan['jobs'] if isinstance(plan,dict) else plan
    selected=[j for j in all_jobs if j['model']==model and (condition is None or j['condition']==condition)]
    expected={j['job_key']:j for j in selected}
    if not selected or len(expected)!=len(selected):raise Blocked('Empty or duplicate expected jobs')
    if any(j['uid'] not in by_uid for j in selected):raise Blocked('Expected jobs outside selected frozen split')
    natural=all(j.get('stage')=='natural_certification' for j in selected)
    if split=='calibration' and not natural:raise Blocked('Calibration pipeline requires natural certification jobs')
    if split=='main' and any(j.get('stage') not in ('main','controls','rescue','stop') for j in selected):
        raise Blocked('Research pipeline received nonresearch jobs')
    provenance.update(records_sha256=file_digest(records_path),jobs_sha256=file_digest(jobs_path),
                      tokenizer=tokenizer_provenance,
                      module_hashes={name:digest(Path(__file__).with_name(name).read_bytes()) for name in
                                     ('offline_pipeline.py','enrich_records.py','token_partitions.py','task_scoring.py','natural_certification.py')})
    out.mkdir(parents=True)
    seen=set();failures=[];compact=[];counts=Counter()
    with (out/'enriched_scored.jsonl').open('x',encoding='utf-8') as destination, records_path.open(encoding='utf-8') as source:
        for line in source:
            if not line.strip():continue
            record=json.loads(line);key=record['job_key']
            if key in seen:raise Blocked('Duplicate raw job; partial output is not a completion receipt')
            seen.add(key)
            if key not in expected:raise Blocked('Unexpected raw job; do not silently filter observations')
            try:
                result=process_record(record,by_uid[record['uid']],labels[record['uid']],expected[key],tokenizer)
                destination.write(json.dumps(result,ensure_ascii=False)+'\n')
                counts[result['condition']]+=1
                fields=('model','condition','uid','repeat','side','task','image_sha256','seed','record_complete',
                        'output_cap_reached','output_tokens','score','prompt_sha256','reasoning_tokens','think_ended','boundary_anomaly')
                compact.append({f:result[f] for f in fields if f in result})
            except (Blocked,KeyError,ValueError,TypeError) as exc:
                failures.append({'job_key':key,'status':'blocked','score':None,'error':str(exc)[:1000]})
    missing=sorted(set(expected)-seen)
    complete=not missing and not failures
    certification={}
    if natural:
        from .natural_certification import evaluate
        for name in sorted({j['condition'] for j in selected}):
            subset=[r for r in compact if r['condition']==name]
            try:certification[name]=evaluate(subset,manifest,model,name)
            except Blocked as exc:certification[name]={'status':'INCONCLUSIVE','reason':str(exc),'native_gate_passed':False}
    report={'records_complete':complete,'expected_records':len(expected),'raw_records':len(seen),
            'scored_records':len(compact),'per_condition_scored':dict(counts),'missing_job_keys':missing,
            'failures':failures,'natural_certification':certification,'provenance':provenance,
            'native_gate_passed':False,'production_ready':False,
            'scope':'Offline partition/scoring/statistics only; completeness is not native or engine certification.'}
    report['enriched_scored_sha256']=file_digest(out/'enriched_scored.jsonl')
    publish_once(out/'completion.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('records','frozen','jobs','tokenizer','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=['main','calibration'],required=True)
    p.add_argument('--model',choices=['llava7','mini8','qwen8','qwen4'],required=True)
    p.add_argument('--condition')
    a=p.parse_args()
    if not a.tokenizer.is_dir():raise Blocked('Use a local frozen tokenizer directory')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(a.tokenizer),use_fast=a.model.startswith('qwen'),local_files_only=True,trust_remote_code=False)
    names=('tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt')
    provenance={'path':str(a.tokenizer.resolve()),'class':type(tokenizer).__name__,
                'files':{name:file_digest(a.tokenizer/name) for name in names if (a.tokenizer/name).exists()}}
    result=run_pipeline(a.records,a.frozen,a.split,a.jobs,a.out,a.model,tokenizer,provenance,a.condition)
    print(json.dumps({k:result[k] for k in ('records_complete','expected_records','scored_records','native_gate_passed')}))


if __name__=='__main__':main()
