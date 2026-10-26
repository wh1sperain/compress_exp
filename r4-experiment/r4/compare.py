"""Compare same-prefix logits. Report evidence, never silently choose a tolerance."""
import argparse
import numpy as np
from pathlib import Path
from .core import read,write_once

def compare(left,right):
    a,b=Path(left),Path(right)
    if (a/'greedy32.json').exists():
        ra,rb=read(a/'greedy32.json'),read(b/'greedy32.json')
        for k in ('uid','image_sha256','prompt_sha256'):
            if ra[k]!=rb[k]:raise ValueError('Greedy inputs differ')
        return dict(greedy_equal=ra['generated_token_ids']==rb['generated_token_ids'],gate_passed=False,
                    reference_tokens=len(ra['generated_token_ids']),candidate_tokens=len(rb['generated_token_ids']))
    ra,rb=read(a/'receipt.json'),read(b/'receipt.json')
    for k in ('uid','image_sha256','prefix_sha256','tokens','captured_steps'):
        if ra[k]!=rb[k]:raise ValueError('Unpaired numeric probe '+k)
    x,y=np.load(a/'logits.npz'),np.load(b/'logits.npz');rows=[]
    for k in x.files:
        u,v=x[k].astype(float),y[k].astype(float)
        if not np.isfinite(u).all() or not np.isfinite(v).all():raise ValueError('Nonfinite logits')
        p=np.exp(u-u.max());p/=p.sum();q=np.exp(v-v.max());q/=q.sum()
        rows.append(dict(step=int(k),max_abs=float(abs(u-v).max()),argmax_equal=bool(u.argmax()==v.argmax()),
            tv=float(abs(p-q).sum()/2),kl=float((p*(np.log(p+1e-300)-np.log(q+1e-300))).sum())))
    return dict(comparisons=rows,gate_passed=False,requires_calibrated_baseline=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();write_once(a.out,compare(a.left,a.right))
