"""Real-image diagnostics, never research generation or a passing native gate.

Shared teacher-forcing prefixes permit cross-environment Full comparisons.
Author independence must be established separately; self-observation is not proof.
"""
import time
from pathlib import Path
from .common import write, digest, Blocked, code_hash


def cache_lengths(cache):
    if cache is None:
        raise Blocked('Requested cache absent')
    if hasattr(cache, 'key_cache'):
        return [int(k.shape[-2]) for k in cache.key_cache]
    if hasattr(cache, 'layers'):
        return [int(layer.keys.shape[-2]) for layer in cache.layers]
    return [int(layer[0].shape[-2]) for layer in cache]


def run(model, inputs, tokenizer, layers, out, provenance, natural_cap):
    import torch
    from transformers import LogitsProcessorList
    out=Path(out)
    if out.exists():raise Blocked('Diagnostic attempt already exists')
    out.mkdir(parents=True)
    records=[];snapshots={};handles=[];step=[0]
    probes={0,1,127,128,511,512,1023,1024}
    prefix=tokenizer.encode(' The image contains information that can be examined carefully.',add_special_tokens=False)
    if not prefix:raise Blocked('Empty forcing prefix')
    forced=(prefix* (1025//len(prefix)+1))[:1025]
    write(out/'identity.json',dict(provenance,code_sha256=code_hash(),teacher_prefix_sha256=digest(str(forced)),
        teacher_steps=1025,natural_cap=natural_cap,production_ready=False))
    def pre(index):
        def hook(module,args,kwargs):
            hidden=args[0] if args else kwargs['hidden_states']
            item={'step':step[0],'layer':index,'hidden_shape':list(hidden.shape)}
            for name in ('position_ids','cache_position','attention_mask'):
                value=kwargs.get(name)
                item[name]=None if value is None else {'shape':list(value.shape),'dtype':str(value.dtype)}
                if value is not None and (name!='attention_mask' or step[0] in probes):
                    # Small positions are kept exactly; large prefill masks retain range.
                    if name!='attention_mask':item[name]['values']=value.detach().cpu().tolist()
                    elif value.numel():item[name].update(min=float(value.min()),max=float(value.max()))
            records.append(item)
        return hook
    for i,layer in enumerate(layers):handles.append(layer.register_forward_pre_hook(pre(i),with_kwargs=True))
    kv=[]
    def post(module,args,output):
        lengths=cache_lengths(output.past_key_values)
        if len(lengths)!=len(layers):raise Blocked('Incomplete layer cache')
        kv.append({'step':step[0],'lengths':lengths})
    handles.append(model.register_forward_hook(post))
    class Force:
        def __call__(self,ids,scores):
            i=step[0]
            if i in probes:snapshots[i]=scores[0].detach().float().cpu()
            result=torch.full_like(scores,-float('inf'));result[:,forced[i]]=0
            step[0]+=1
            if step[0]%128==0:write(out/'status.json',{'stage':'teacher_forcing','completed':step[0],'updated_at':time.time()})
            return result
    try:
        with torch.inference_mode():
            model.generate(**inputs,max_new_tokens=1025,do_sample=False,use_cache=True,
                eos_token_id=None,forced_eos_token_id=None,logits_processor=LogitsProcessorList([Force()]))
        if step[0]!=1025 or len(kv)!=1025:raise Blocked('Teacher forcing ended early')
        # Each already-established layer cache must grow by one per decode step.
        for previous,current in zip(kv,kv[1:]):
            if any(b!=a+1 for a,b in zip(previous['lengths'],current['lengths'])):
                raise Blocked('Non-unit cache growth')
        write(out/'teacher_status.json',{'status':'PASS','steps':step[0],'scope':'cache growth only; numerical equivalence pending'})
    except BaseException as exc:
        write(out/'teacher_status.json',{'status':'FAIL','steps':step[0],'error_type':type(exc).__name__,'error':str(exc)[:1200]})
        raise
    finally:
        for handle in handles:handle.remove()
        write(out/'layer_inputs.json',records);write(out/'layer_kv.json',kv)
        torch.save(snapshots,out/'teacher_logits.pt')
    from .reference_stream import generate_native
    write(out/'status.json',{'stage':'natural_generation','updated_at':time.time()})
    result=generate_native(model,inputs,tokenizer,
        {'job_key':'validation-'+provenance['method'],'seed':1234,
         'generation':dict(max_new_tokens=natural_cap,temperature=1.,top_k=20,top_p=.95)},out/'natural.prefix.json')
    write(out/'natural.json',result)
    write(out/'status.json',{'stage':'complete','status':'INCONCLUSIVE','production_ready':False,
        'reason':'Independent author comparison and full gate still required','updated_at':time.time()})
