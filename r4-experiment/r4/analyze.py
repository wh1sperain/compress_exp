"""Pair complete questions; stratified image-cluster bootstrap. Missing stays missing."""
import argparse
from collections import defaultdict
from .core import *

def paired(rows,condition_name,task,field,draws=20000,seed_value=20261026,family=210):
    import numpy as np
    by=defaultdict(dict)
    for r in rows:
        if r['task']==task and r['condition'] in ('Full',condition_name):
            key=(r['uid'],r['repeat'])
            if r['condition'] in by[key]:raise ValueError('Duplicate paired record')
            by[key][r['condition']]=r
    questions=defaultdict(list);image={};missing=0
    for (uid,repeat),pair in by.items():
        if set(pair)!=set(('Full',condition_name)):raise ValueError('Incomplete condition pairing')
        a,b=pair['Full'],pair[condition_name]
        if a['seed']!=b['seed'] or a['image_sha256']!=b['image_sha256']:raise ValueError('Pair identity mismatch')
        if a.get(field) is None or b.get(field) is None:
            missing+=1;continue
        questions[uid].append((repeat,float(b[field])-float(a[field])))
        image[uid]=a['image_sha256']
    values={u:sum(v for _,v in x)/3 for u,x in questions.items() if {r for r,_ in x}=={0,1,2}}
    incomplete=len(set(u for u,_ in by)-set(values))
    if not values:return dict(task=task,condition=condition_name,field=field,status='INCOMPLETE',missing_pairs=missing)
    clusters=defaultdict(list)
    for u,v in values.items():clusters[image[u]].append(v)
    sums=np.array([sum(v) for v in clusters.values()]);counts=np.array([len(v) for v in clusters.values()])
    rng=np.random.Generator(np.random.PCG64(seed_value));boot=[]
    for _ in range(draws):
        i=rng.integers(0,len(sums),len(sums));boot.append(float(sums[i].sum()/counts[i].sum()))
    alpha=.05/family
    return dict(task=task,condition=condition_name,field=field,status='COMPLETE' if not incomplete else 'PARTIAL',
        complete_questions=len(values),incomplete_questions=incomplete,missing_pairs=missing,image_clusters=len(clusters),
        mean_difference=sum(values.values())/len(values),ci95=np.quantile(boot,[.025,.975]).tolist(),
        family_ci95=np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),family=family,
        caveat='Partial annotation subset is not population error prevalence; capped lengths remain observed/censored')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();rows=read(a.records)
    write_once(a.out,[paired(rows,c,t,f) for t in TASKS for c in CONDITIONS[1:]
                      for f in ('reasoning_tokens','quality','visual_error')])
