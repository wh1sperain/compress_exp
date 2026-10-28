"""Paired question bootstrap and within-task macro bootstrap; no seed pseudoreplication."""
import math
import numpy as np

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]

def ctir(base,compressed):
    den=float(np.sum(base))
    return 100*(float(np.sum(compressed))-den)/den if den else None

def bootstrap(base,compressed,base_correct=None,compressed_correct=None,n=10000,seed=20261027):
    b=np.asarray(base,dtype=float);c=np.asarray(compressed,dtype=float)
    if len(b)!=len(c) or not len(b):raise ValueError('Empty/unpaired bootstrap')
    rng=np.random.default_rng(seed);ct=[];acc=[];delta=[]
    for start in range(0,n,256):
        ix=rng.integers(0,len(b),size=(min(256,n-start),len(b)))
        den=b[ix].sum(axis=1);num=c[ix].sum(axis=1)
        ct.extend(np.divide(100*(num-den),den,out=np.full(len(ix),np.nan),where=den!=0))
        delta.extend((c[ix]-b[ix]).mean(axis=1))
        if base_correct is not None:
            acc.extend(100*(np.asarray(compressed_correct)[ix]-np.asarray(base_correct)[ix]).mean(axis=1))
    def ci(v):return np.percentile(v,[2.5,97.5]).tolist() if len(v) and np.isfinite(v).all() else None
    return dict(ctir_ci95=ci(ct),delta_tokens_ci95=ci(delta),accuracy_delta_pp_ci95=ci(acc),
                undefined_bootstrap_count=int(np.isnan(ct).sum()),resamples=n,unit='sample_id',seed=seed)

def macro_bootstrap(task_pairs,n=10000,seed=20261027):
    if len(task_pairs)!=6:raise ValueError('Macro requires all six tasks')
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,n,256):
        values=[]
        for b,c in task_pairs:
            b=np.asarray(b);c=np.asarray(c);ix=rng.integers(0,len(b),size=(min(256,n-start),len(b)))
            den=b[ix].sum(axis=1)
            values.append(np.divide(100*(c[ix].sum(axis=1)-den),den,out=np.full(len(ix),np.nan),where=den!=0))
        samples.extend(np.mean(values,axis=0))
    return np.percentile(samples,[2.5,97.5]).tolist() if np.isfinite(samples).all() else None

def bh(pvalues):
    """Exploratory FDR only; accepts prespecified valid p-values, not CI-derived guesses."""
    order=np.argsort(pvalues);q=np.empty(len(order));running=1.
    for i in range(len(order)-1,-1,-1):
        running=min(running,pvalues[order[i]]*len(order)/(i+1));q[order[i]]=running
    return q.tolist()
