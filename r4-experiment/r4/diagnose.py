"""Teacher-forced probe of a REAL input and supplied prefix; stores raw numeric evidence.

Run native and wrapper in isolated pinned environments with identical prefix IDs.
This never creates a PASS gate. Natural 20-image tests use r4.run --diagnostic.
"""
import argparse
import sys
from pathlib import Path
from .core import *
from .run import identity,R3

def probe(session,row,job,prefix,out):
    import torch
    import numpy as np
    from transformers import LogitsProcessorList
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    batch,extra=session.prepare(row,job)
    steps={0,31,1023,4095,len(prefix)-1};captured={};index=[0]
    def force(ids,scores):
        i=index[0]
        if i in steps:captured[str(i)]=scores[0].detach().float().cpu().numpy()
        result=torch.full_like(scores,-torch.inf);result[0,prefix[i]]=0;index[0]+=1
        return result
    with torch.inference_mode():
        session.model.generate(**batch,**extra,do_sample=False,max_new_tokens=len(prefix),
            eos_token_id=[],forced_eos_token_id=None,logits_processor=LogitsProcessorList([force]),use_cache=True)
    if index[0]!=len(prefix):raise ValueError('Teacher forcing ended early')
    np.savez_compressed(out/'logits.npz',**captured)
    write_once(out/'trace.json',session.trace)
    write_once(out/'receipt.json',dict(uid=row['uid'],image_sha256=row['image_sha256'],
        prefix_sha256=sha(prefix),tokens=len(prefix),captured_steps=sorted(captured),
        native_gate_passed=False,formal=False))

def greedy32(session,row,job,out):
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    batch,extra=session.prepare(row,job)
    with torch.inference_mode():
        output=session.model.generate(**batch,**extra,do_sample=False,max_new_tokens=32,use_cache=True)
    tokens=output[0,batch['input_ids'].shape[1]:].tolist()
    write_once(out/'greedy32.json',dict(uid=row['uid'],image_sha256=row['image_sha256'],
        prompt_sha256=session.trace['prompt_sha256'],generated_token_ids=tokens,native_gate_passed=False))
    write_once(out/'trace.json',session.trace)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('row','job','condition','checkpoint','author-root','weights-lock','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--prefix');p.add_argument('--greedy32',action='store_true')
    p.add_argument('--spec',default='protocol.json');p.add_argument('--locks',default='sources.lock.json')
    p.add_argument('--disabled',action='store_true');p.add_argument('--native',action='store_true')
    p.add_argument('--keep100',action='store_true');a=p.parse_args()
    spec=read(a.spec);locks=read(a.locks);ident=identity(spec,locks,a.checkpoint,a.weights_lock)
    from .adapters import Session
    s=Session(a.condition,a.checkpoint,a.author_root,spec,locks,disabled=a.disabled,native=a.native,keep100=a.keep100)
    if a.greedy32:greedy32(s,read(a.row),read(a.job),a.out)
    else:
        if not a.prefix:raise ValueError('--prefix required for teacher forcing')
        prefix=read(a.prefix)
        if not 32<=len(prefix)<=32767 or any(not isinstance(t,int) or t<0 for t in prefix):raise ValueError('Invalid probe prefix')
        probe(s,read(a.row),read(a.job),prefix,a.out)
    write_once(Path(a.out)/'identity.json',ident)
