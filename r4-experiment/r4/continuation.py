"""Replay exact prefix tokens, then continue with matched request uniforms.

Edited prefixes are explicit interventions, not recovery of natural experiments.
"""
import time
from .core import sha

def generate(session,batch,extra,job,recipe):
    import torch
    from transformers import LogitsProcessorList
    from r3.sampling import uniforms,gpu_sample
    prefix=recipe['forced_prefix'];offset=recipe['rng_offset'];remaining=recipe['max_new_tokens']
    if len(prefix)+remaining>32768 or remaining<1 or offset<0:raise ValueError('Intervention cap/RNG invalid')
    eos=session.model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else eos
    if any(x in (eos or []) for x in prefix):raise ValueError('Do not replay EOS inside thought prefix')
    us=torch.tensor(uniforms(job['seed'],offset+remaining),device=session.model.device)
    ids=[];start=time.perf_counter()
    def sampler(input_ids,scores):
        i=len(ids)
        if i<len(prefix):token=torch.tensor([prefix[i]],device=scores.device)
        else:
            g=job['generation']
            token=gpu_sample(scores,us[offset+i-len(prefix):offset+i-len(prefix)+1],
                temperature=g['temperature'],top_k=g['top_k'],top_p=g['top_p'])
        ids.append(int(token.item()));forced=torch.full_like(scores,-torch.inf)
        forced.scatter_(1,token[:,None],0.);return forced
    with torch.inference_mode():
        session.model.generate(**batch,**extra,do_sample=True,temperature=1.,top_k=0,top_p=1.,
            repetition_penalty=1.,max_new_tokens=len(prefix)+remaining,use_cache=True,
            logits_processor=LogitsProcessorList([sampler]))
    if ids[:len(prefix)]!=prefix:raise ValueError('Forced prefix not preserved')
    return dict(generated_token_ids=ids,output_text=session.processor.tokenizer.decode(ids,skip_special_tokens=False),
        output_tokens=len(ids),forced_prefix_tokens=len(prefix),continuation_tokens=len(ids)-len(prefix),
        rng_offset=offset,recipe_sha256=sha(recipe),wall_seconds=time.perf_counter()-start,
        output_cap_reached=len(ids)==32768,eos_terminated=bool(ids and ids[-1] in (eos or [])),
        interpretation='intervention; not natural generation or equivalent natural RNG sequence')
