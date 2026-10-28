"""Freeze representative paired cases, keeping original question/image and BOTH traces."""
import argparse
from collections import defaultdict
from .common import *

def select_cases(records,inputs,limit=60):
    if not 40<=limit<=60:raise ValueError('Select 40-60 representative pairs')
    meta={r['sample_id']:r for r in inputs};base={}
    def key(r):return (r['phase'],r['model'],r['sample_id'],r['decoding_mode'],r['seed'])
    for r in records:
        if r['method']=='Full':base[key(r)]=r
    groups=defaultdict(list)
    for r in records:
        if r['method']=='Full' or key(r) not in base:continue
        b=base[key(r)];bl=b.get('reasoning_tokens');cl=r.get('reasoning_tokens')
        if bl is None or cl is None or bl<32:continue
        ratio=cl/bl;quality=None
        if type(b.get('correct')) is bool and type(r.get('correct')) is bool:
            quality=('C' if b['correct'] else 'W')+('C' if r['correct'] else 'W')
        category='shortened' if ratio<=.5 else quality if ratio>=2 and quality in ('CC','CW','WW') else None
        if category:
            groups[(r['dataset'],r['method'],r['model'],category)].append(dict(input=meta[r['sample_id']],baseline=b,compressed=r,
                ratio=ratio,delta_tokens=cl-bl,category=category,review=None,
                review_dimensions=['rechecking','backtracking','unproductive_branches','recalculation','visual_uncertainty','failure_to_stop']))
    for k in groups:groups[k].sort(key=lambda x:digest([42,x['compressed']['run_id']]))
    chosen=[];used=set()
    while len(chosen)<limit:
        progress=False
        for k in sorted(groups):
            choices=[x for x in groups[k] if x['compressed']['run_id'] not in used]
            if choices:
                x=choices[0];chosen.append(x);used.add(x['compressed']['run_id']);progress=True
                if len(chosen)==limit:break
        if not progress:break
    return dict(cases=chosen,count=len(chosen),target=limit,shortage=max(0,limit-len(chosen)),
                scope='stratified qualitative sample; frequencies not population estimates; human review pending')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--inputs',required=True);p.add_argument('--out',required=True)
    p.add_argument('--limit',type=int,default=60);a=p.parse_args();save(a.out,select_cases(rows(a.records),rows(a.inputs),a.limit))
