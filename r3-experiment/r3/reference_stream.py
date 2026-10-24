"""Request-local native HF sampling and immutable prefix checkpoints.

The native generation loop retains its position/cache/stop policy. This logits
processor supplies the protocol's single sampled token on each step. Built-in
temperature/top-k/top-p must be neutralized to avoid applying them twice.
"""
from __future__ import annotations
from pathlib import Path
import time
from .common import read,write,Blocked
from .sampling import uniforms,gpu_sample


class RequestSampler:
    def __init__(self,job,checkpoint,device):
        import torch
        self.job=job;self.checkpoint=Path(checkpoint);self.step=0
        self.u=torch.tensor(uniforms(job['seed'],job['generation']['max_new_tokens']),device=device)
        self.saved=[];self.ids=[]
        if self.checkpoint.exists():
            receipt=read(self.checkpoint)
            if receipt['job_key']!=job['job_key']:raise Blocked('Checkpoint belongs to another job')
            if receipt['seed']!=job['seed']:raise Blocked('Checkpoint RNG seed changed')
            if receipt['generated_steps']!=len(receipt['generated_token_ids']):raise Blocked('RNG/prefix offset mismatch')
            self.saved=receipt['generated_token_ids']
        self.start=time.time()

    def __call__(self,input_ids,scores):
        import torch
        if scores.shape[0]!=1:raise Blocked('Reference sampler is B1 only')
        if self.step>=len(self.u):raise Blocked('Native generator exceeded protocol cap')
        g=self.job['generation']
        token=gpu_sample(scores,self.u[self.step:self.step+1],temperature=g['temperature'],top_k=g['top_k'],top_p=g['top_p'])
        if self.step<len(self.saved):
            # Rebuild from original prefix, never resample a failed job. Refuse
            # if the unchanged backend no longer agrees with the saved RNG.
            if token.item()!=self.saved[self.step]:raise Blocked('Prefix reconstruction disagrees with saved sample')
        self.ids.append(token)
        self.step+=1
        if self.step%128==0:self.flush()
        forced=torch.full_like(scores,-float('inf'))
        forced.scatter_(1,token[:,None],0.)
        return forced

    def flush(self):
        import torch
        ids=torch.cat(self.ids).tolist() if self.ids else []
        from .prefix_checkpoint import publish
        publish(self.checkpoint,{'job_key':self.job['job_key'],'generated_token_ids':ids,
                              'generated_steps':len(ids),'seed':self.job['seed'],
                              'updated_at':time.time(),'rebuild_prefix_tokens':len(self.saved)},self.saved)
        return ids


def generate_native(model,inputs,tokenizer,job,checkpoint,extra=None):
    import torch
    from transformers import LogitsProcessorList
    sampler=RequestSampler(job,checkpoint,model.device)
    kwargs=dict(do_sample=True,temperature=1.,top_k=0,top_p=1.,repetition_penalty=1.,
                min_new_tokens=0,max_new_tokens=job['generation']['max_new_tokens'],use_cache=True,
                logits_processor=LogitsProcessorList([sampler]))
    if extra:kwargs.update(extra)
    with torch.inference_mode():model.generate(**inputs,**kwargs)
    ids=sampler.flush()
    if len(ids)<len(sampler.saved):raise Blocked('Generation ended before saved prefix was fully reconstructed')
    eos=model.generation_config.eos_token_id
    eos=[] if eos is None else [eos] if isinstance(eos,int) else eos
    ended=bool(ids and ids[-1] in eos)
    return {'generated_token_ids':ids,'output_tokens':len(ids),
            'output_text':tokenizer.decode(ids,skip_special_tokens=False),
            'eos_terminated':ended,'output_cap_reached':len(ids)==job['generation']['max_new_tokens'],
            'wall_seconds':time.time()-sampler.start,'reconstructed_prefix_tokens':len(sampler.saved)}
