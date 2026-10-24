"""R3 paired analysis of complete, externally scored research rows only.

Never performs inference, fills missing observations, or silently drops caps.
Repeated seeds are averaged within image before within-task cluster bootstrap.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .common import read,write,Blocked
from .manifest import TASKS


def paired(rows,model,condition,endpoint):
    selected=[r for r in rows if r['model']==model and r['stage']=='main' and r['condition'] in ('Full',condition)]
    index={}
    for r in selected:
        key=(r['uid'],r['repeat'],r['condition'])
        if key in index:raise Blocked('Duplicate paired observation: '+str(key))
        index[key]=r
    grouped={}
    for uid in sorted({r['uid'] for r in selected}):
        repeats=[]
        for repeat in range(3):
            try:f=index[(uid,repeat,'Full')];c=index[(uid,repeat,condition)]
            except KeyError as e:raise Blocked('Incomplete repeated pair: '+str(e))
            for field in ('task','image_sha256','seed','prompt_sha256'):
                if f[field]!=c[field]:raise Blocked('Pair mismatch '+field+': '+uid)
            if f[endpoint] is None or c[endpoint] is None:raise Blocked('Missing primary length: '+uid)
            lf,lc=f[endpoint],c[endpoint]
            repeats.append([lc-lf,float(lf>0 and lc>2*lf and lc-lf>=256),
                            float(lf>0 and lc>2*lf),float(c['output_cap_reached'])-float(f['output_cap_reached']),
                            float(c['score'])-float(f['score'])])
        image=f['image_sha256'];key=(f['task'],image)
        if key in grouped:raise Blocked('More than one question per sampled image')
        grouped[key]=np.mean(repeats,axis=0)
    return grouped


def intervals(grouped,family_size,replicates=20000,seed=20261022):
    rng=np.random.Generator(np.random.PCG64(seed));task_boot=[];task_point={}
    for task in TASKS:
        values=np.asarray([v for (t,h),v in sorted(grouped.items()) if t==task],dtype=np.float64)
        if len(values)!=100:raise Blocked(task+': expected exactly 100 image clusters')
        samples=np.empty((replicates,values.shape[1]))
        for start in range(0,replicates,250):
            stop=min(start+250,replicates)
            ids=rng.integers(0,len(values),size=(stop-start,len(values)))
            samples[start:stop]=values[ids].mean(axis=1)
        task_boot.append(samples);task_point[task]=values.mean(axis=0).tolist()
    boot=np.mean(task_boot,axis=0);point=np.mean(list(task_point.values()),axis=0)
    ordinary=np.quantile(boot,[.025,.975],axis=0).T
    family=np.quantile(boot,[.025/family_size,1-.025/family_size],axis=0).T
    return {'task_equal_weight_mean':point[:4].tolist(),'per_task_mean':{t:v[:4] for t,v in task_point.items()},
            'per_task_score_difference':{t:v[4] for t,v in task_point.items()},
            'ordinary_95_ci':ordinary[:4].tolist(),'primary_family_95_ci':family[:2].tolist(),
            'family_size':family_size,'replicates':replicates,'seed':seed,
            'fields':['paired_length_difference','severe_over2_probability','over2_probability','paired_cap_difference'],
            'family_ci_scope':'Only the first two endpoints; registered 25 percent comparisons only.',
            'quality_note':'Quality reported separately by task; no pooled raw task score.'}


def validate_complete(rows,expected_jobs):
    expected={r['job_key']:r for r in expected_jobs}
    actual={}
    for row in rows:
        key=row['job_key']
        if key in actual:raise Blocked('Duplicate job_key')
        if key not in expected:raise Blocked('Unexpected research job')
        for field in ('model','uid','repeat','condition','stage','seed','image_sha256'):
            if row[field]!=expected[key][field]:raise Blocked('Job identity mismatch: '+field)
        if not row.get('record_complete'):raise Blocked('Incomplete result record')
        if len(row['generated_token_ids'])!=row['output_tokens']:raise Blocked('Token-count mismatch')
        if row['output_tokens']>expected[key]['generation']['max_new_tokens']:raise Blocked('Output cap exceeded')
        if row['output_cap_reached']!=(row['output_tokens']==expected[key]['generation']['max_new_tokens']):raise Blocked('Incorrect cap flag')
        actual[key]=row
    if set(actual)!=set(expected):raise Blocked(f'Missing {len(set(expected)-set(actual))} registered rows')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True);p.add_argument('--rows',required=True);p.add_argument('--jobs',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();spec=read(a.spec)
    rows=[json.loads(s) for s in Path(a.rows).read_text().splitlines() if s.strip()]
    jobs=read(a.jobs)['jobs'];validate_complete(rows,jobs)
    report={'protocol':spec['protocol'],'research_records':len(rows),'paired':{},
            'limitations':['Capped lengths are censored observations, not natural completion lengths.',
                           'Mini is descriptive; no cross-backbone causal inflation ranking.']}
    for track in spec['main_tracks'][:2]:
        model=track['model'];endpoint='output_tokens' if model=='llava7' else 'reasoning_tokens'
        for condition in track['conditions']:
            if condition=='Full':continue
            family=8 if model=='llava7' else 2
            values=paired(rows,model,condition,endpoint)
            result=intervals(values,family,spec['statistics']['bootstrap_replicates'],spec['statistics']['bootstrap_seed'])
            result['primary']=condition in spec['statistics']['A_primary_methods'] if model=='llava7' else condition=='PriorTR25'
            result['length_endpoint']=endpoint
            report['paired'][model+'/'+condition]=result
    write(a.out,report)


if __name__=='__main__':main()
