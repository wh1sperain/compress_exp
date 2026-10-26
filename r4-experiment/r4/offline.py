"""Join raw immutable results, frozen extraction scores and human facts locally."""
import argparse
import sys
from pathlib import Path
from .core import *
from .annotations import validate
sys.path.insert(0,str(ROOT.parent/'r3-experiment'))

def collect(result_root,panel,tokenizer,quality=None,annotations=None):
    from r3.token_partitions import partition_generated,prompt_has_open_think
    q={r['job_key']:r for r in (quality or [])};a={r['job_key']:r for r in (annotations or [])}
    if len(q)!=len(quality or []) or len(a)!=len(annotations or []):raise ValueError('Duplicate joined record')
    rows={r['uid']:r for r in panel};out=[];seen=set()
    for p in sorted(Path(result_root).rglob('result.json')):
        r=read(p)
        if not r.get('formal'):continue
        job=r['job'];k=job['job_key']
        if k in seen:raise ValueError('Duplicate success '+k)
        seen.add(k);trace=read(p.with_name('trace.json'))
        if sha(trace)!=r['trace_sha256']:raise ValueError('Trace changed')
        row=rows[job['uid']]
        if job['input_sha256']!=sha(input_only(row)):raise ValueError('Input changed')
        ids=r['generated_token_ids']
        part=partition_generated(ids,tokenizer,'qwen8',32768,r['generation_eos_token_id'],
            prompt_think_open=prompt_has_open_think(trace['prompt_ids'],tokenizer))
        score=q.get(k)
        if score and (score['output_sha256']!=sha(ids) or not score.get('extraction_audit_passed')):
            raise ValueError('Quality score not bound to reviewed extraction/output')
        visual_error=None;first=None
        if k in a:
            ar=a[k]
            if ar['output_sha256']!=sha(ids) or ar['image_sha256']!=row['image_sha256']:raise ValueError('Annotation binding changed')
            first=validate(ar,len(ids));visual_error=ar['visual_error']
        boundary=part['thinking_boundary_offset']
        thought_end=len(ids) if boundary is None else boundary
        if first is not None and first>=thought_end:raise ValueError('First reasoning error annotated in final answer')
        out.append(dict(job_key=k,uid=job['uid'],task=job['task'],repeat=job['repeat'],seed=job['seed'],
            condition=job['condition'],image_sha256=row['image_sha256'],**part,
            quality=None if score is None else score['quality'],visual_error=visual_error,
            first_error_output_offset=first,after_error_observed_tokens=None if first is None else sum(t not in tokenizer.all_special_ids for t in ids[first:thought_end]),
            quality_protocol=None if score is None else score['protocol_sha256']))
    return out

def descriptive(rows):
    import numpy as np
    report=[]
    full={(r['uid'],r['repeat']):r for r in rows if r['condition']=='Full'}
    for task in TASKS:
        for c in CONDITIONS:
            rs=[r for r in rows if r['task']==task and r['condition']==c]
            if not rs:continue
            ll=[r['reasoning_tokens'] for r in rs]
            pairs=[(r,full[(r['uid'],r['repeat'])]) for r in rs if (r['uid'],r['repeat']) in full]
            report.append(dict(task=task,condition=c,n=len(rs),mean_observed_reasoning=float(np.mean(ll)),
                median=float(np.median(ll)),p95=float(np.quantile(ll,.95)),
                cap_count=sum(r['output_cap_reached'] for r in rs),
                censored_count=sum(r['output_right_censored'] for r in rs),
                unclosed_count=sum(not r['thinking_boundary_found'] for r in rs),
                paired_n=len(pairs),growth_ge2_count=sum(r['output_tokens']>=2*f['output_tokens'] for r,f in pairs),
                quality_scored_n=sum(r['quality'] is not None for r in rs),human_reviewed_n=sum(r['visual_error'] is not None for r in rs)))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('results','panel','checkpoint','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--quality');p.add_argument('--annotations');a=p.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True)
    rows=collect(a.results,read(a.panel),tokenizer,read(a.quality) if a.quality else None,read(a.annotations) if a.annotations else None)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    write_once(out/'records.json',rows);write_once(out/'descriptive.json',descriptive(rows))
