"""Generate explicit matrices; greedy fallback blocks formal scheduling until resolved."""
import argparse
from .common import *

def matrix(panel, spec, phase, decisions=None, fallback_plan=None):
    if phase not in ('precheck','pilot','formal','robustness','determinism'): raise ValueError('Unknown phase')
    if phase=='formal':
        if not decisions or any(decisions.get(m,{}).get('mode') not in ('greedy','sampling') for m in MODELS):raise ValueError('Need both precheck decisions')
        if any(decisions[m]['mode']=='sampling' for m in MODELS):
            if not fallback_plan or fallback_plan.get('decisions_sha256')!=digest(decisions) or fallback_plan.get('core_seeds')!=[42,43,44]:
                raise ValueError('Sampling fallback requires a separately frozen multiseed plan bound to precheck')
    if phase=='formal': expected=150
    elif phase in ('precheck','pilot'): expected=20
    else: expected=None
    if expected and any(sum(r['dataset']==d for r in panel)!=expected for d in DATASETS): raise ValueError('Wrong panel size')
    if phase=='robustness':
        panel=[r for d in ('MathVista','ChartQA') for r in sorted((x for x in panel if x['dataset']==d),key=lambda x:digest([42,x['sample_id']]))[:60]]
        if len(panel)!=120: raise ValueError('Need 60 formal questions per robustness task')
    if phase=='determinism':
        panel=sorted(panel,key=lambda x:digest([42,x['sample_id']]))[:10]
        if len(panel)!=10: raise ValueError('Need 10 independent pilot questions')
    conditions=[('Full',100)]
    if phase in ('pilot','formal','determinism'):
        conditions+= [(m,r) for r in (RATIOS if phase=='formal' else (75,50,25)) for m in METHODS]
    if phase=='determinism': conditions=conditions[1:]
    if phase=='robustness': conditions += [(m,r) for r in (50,25) for m in ('PriorTR','FastV')]
    jobs=[]
    for method,ratio in conditions:  # BOTH models' Full precede every compressed condition
        for model in MODELS:
            mode='sampling' if phase=='robustness' else 'greedy'
            if phase in ('pilot','formal') and decisions: mode=decisions[model]['mode']
            for row in panel:
                multiseed=phase=='robustness' or (phase=='formal' and mode=='sampling' and (method=='Full' or method in ('PriorTR','FastV') and ratio in (50,25)))
                for seed in ((42,43,44) if multiseed else (42,)):
                    for repeat in (range(3) if phase=='determinism' else (0,)):
                        j=dict(phase=phase,sample_id=row['sample_id'],dataset=row['dataset'],model=model,
                               model_id=spec['models'][model]['model_id'],method=method,keep_ratio_target=ratio/100,
                               condition=method if method=='Full' else method+str(ratio),seed=seed,repeat=repeat,
                               decoding_mode=mode,decoding_config=generation(mode),input_sha256=digest(input_only(row)),
                               protocol_sha256=digest(spec),decision_sha256=digest(decisions) if decisions else None,
                               fallback_plan_sha256=digest(fallback_plan) if fallback_plan else None)
                        j['run_id']=digest(j);jobs.append(j)
    return jobs

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--phase',required=True)
    p.add_argument('--decisions');p.add_argument('--fallback-plan');p.add_argument('--spec',default='extension_protocol.json');p.add_argument('--out',required=True)
    a=p.parse_args();save_rows(a.out,matrix(rows(a.inputs),read(a.spec),a.phase,read(a.decisions) if a.decisions else None,read(a.fallback_plan) if a.fallback_plan else None))
