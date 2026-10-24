"""Register immutable conditions; dry-run never certifies unresolved native inputs."""
import argparse,json
from pathlib import Path
from .common import read,write,digest,seed,Blocked
from .manifest import TASKS

def panel(rows,prefix,per_task,tasks=TASKS):
    return [r for t in tasks for r in sorted([r for r in rows if r['task']==t],key=lambda r:digest(prefix+r['uid']))[:per_task]]

def build(spec,rows,mechanism,profile_sha):
    if len(rows)!=500 or len({r['uid'] for r in rows})!=500 or len({r['image_sha256'] for r in rows})!=500:raise Blocked('Need 500 distinct questions/images')
    if any(sum(r['task']==t for r in rows)!=100 for t in TASKS):raise Blocked('Need 100 images/task')
    result=[]
    panels={'P100':panel(rows,'r3-panel|',20),'B100':panel(rows,'r3-bridge|',50,TASKS[:2])}
    def append(stage,model,rs,conditions,group):
        for repeat in range(3):
            for qi,row in enumerate(sorted(rs,key=lambda r:(TASKS.index(r['task']),r['image_sha256']))):
                cs=conditions[qi%len(conditions):]+conditions[:qi%len(conditions)]
                for condition in cs:
                    temp=.6 if condition.endswith('_T0p6') else 1.
                    base=condition.removesuffix('_T0p6')
                    cfg=dict(spec['conditions'].get(base,{}))
                    if base=='FastV75':cfg={'method':'FastV','keep_ratio':.75,'target_visual_tokens_llava':432,'prune_after_layers':2}
                    if base=='FastV37p5':cfg={'method':'FastV','keep_ratio':.375,'target_visual_tokens_llava':216,'prune_after_layers':2}
                    if base=='Random25_L2':cfg={'method':'Random','keep_ratio':.25,'target_visual_tokens_llava':144,'prune_after_layers':2,'mask_seed':int(digest('r3-mask|'+row['uid'])[:16],16)}
                    if stage in {'rescue','stop'}:cfg={'method':base,'base_condition':'PriorTR25','keep_ratio':.25,'prune_after_layers':3}
                    generation=dict(spec['generation']);generation.update(temperature=temp,max_new_tokens=spec['models'][model]['max_new_tokens'])
                    if stage=='stop':cfg['forced_reasoning_budget']=int(base.split('_')[-1]);cfg['answer_budget']=128
                    identity={'protocol':spec['protocol'],'model':model,'uid':row['uid'],'repeat':repeat,'condition':condition,'config':cfg,'temperature':temp,'backend_profile_hash':profile_sha}
                    result.append(dict(identity,job_key=digest(json.dumps(identity,sort_keys=True)),stage=stage,group=group,task=row['task'],image_sha256=row['image_sha256'],seed=seed(row['uid'],repeat),generation=generation))
    for t in spec['main_tracks']:append('main',t['model'],rows,t['conditions'],t['id'])
    for t in spec['controls']:append('controls',t['model'],panels[t['panel']],t['conditions'],t['id'])
    if mechanism is not None:
        if len(mechanism)!=60 or any(sum(r['task']==t for r in mechanism)!=30 for t in TASKS[:2]):raise Blocked('Need frozen M60, 30/task')
        if not {r['uid'] for r in mechanism}<={r['uid'] for r in rows}:raise Blocked('M60 outside main')
        append('rescue','qwen8',mechanism,spec['mechanism']['new_rescue_arms'],'M60')
        append('stop','qwen8',mechanism,['Stop_'+str(b) for b in spec['mechanism']['forced_reasoning_budgets']],'M60')
    counts={s:sum(r['stage']==s for r in result) for s in ('main','controls','rescue','stop')}
    assert counts['main']==19500 and counts['controls']==2400
    if mechanism is not None:assert counts['rescue']==720 and counts['stop']==540
    assert len({r['job_key'] for r in result})==len(result)
    return result,counts

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--manifest');p.add_argument('--mechanism-manifest');p.add_argument('--engine-profile');p.add_argument('--out',default='jobs.preview.json');m=p.add_mutually_exclusive_group(required=True);m.add_argument('--dry-run',action='store_true');m.add_argument('--production',action='store_true');a=p.parse_args();s=read(a.spec)
    if a.production:raise Blocked('Production job registration awaits native adapter audit and resolved input locks; no research jobs emitted')
    if not a.manifest:
        write(a.out,{'production_ready':False,'counts':s['expected_records'],'requires':['validated main manifest','M60 dual annotations','native and engine gates']});return
    rows=[json.loads(x) for x in Path(a.manifest).read_text(encoding='utf-8').splitlines()]
    mechanism=[json.loads(x) for x in Path(a.mechanism_manifest).read_text(encoding='utf-8').splitlines()] if a.mechanism_manifest else None
    jobs,counts=build(s,rows,mechanism,digest(Path(a.engine_profile).read_bytes()) if a.engine_profile else 'UNRESOLVED')
    write(a.out,{'production_ready':False,'counts':counts,'jobs':jobs})
if __name__=='__main__':main()
