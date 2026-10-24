"""Stable top-k/top-p inverse CDF with identical supplied uniforms."""
import numpy as np

def uniforms(seed,count): return np.random.Generator(np.random.PCG64(seed)).random(count)

def cpu_sample(logits,u,temperature=1.,top_k=20,top_p=.95):
    x=np.asarray(logits,dtype=np.float64)
    if not np.isfinite(x).all() or not 0<=u<1 or temperature<=0:raise ValueError('Invalid sampler input')
    order=np.argsort(-x,kind='stable')[:top_k];v=x[order]/temperature
    probs=np.exp(v-v.max());probs/=probs.sum()
    keep=np.cumsum(probs)-probs<top_p
    probs=np.where(keep,probs,0);probs/=probs.sum()
    return int(order[min(np.searchsorted(np.cumsum(probs),u,side='right'),len(order)-1)])

def gpu_sample(logits,u,temperature=1.,top_k=20,top_p=.95):
    import torch
    if logits.ndim!=2 or u.shape!=(logits.shape[0],):raise ValueError('Expected batched logits/uniforms')
    # Stable sort preserves ascending token ID for equal logits.
    order=torch.argsort(logits,dim=-1,descending=True,stable=True)[:,:top_k]
    values=torch.gather(logits,1,order).double()/temperature
    probs=torch.softmax(values,dim=-1);keep=probs.cumsum(-1)-probs<top_p
    probs=probs*keep;probs=probs/probs.sum(-1,keepdim=True)
    idx=(u[:,None]>=probs.cumsum(-1)).sum(-1).clamp_max(top_k-1)
    return order.gather(1,idx[:,None]).squeeze(1)
