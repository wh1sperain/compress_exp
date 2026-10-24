"""Sequential Qwen teacher forcing; original image/question is pruned only once.

No appended-prefill shortcut: it would alter PriorTR P and TopK. Diagnostic EOS
is disabled, not research EOS. The default remains a 1025-step short diagnostic;
--long-probes requests windows at 1024/4096/8192/16384/32768 generated tokens.
No completion-time promise or automatic native gate is made.
"""
import argparse
import copy
import time
from pathlib import Path
from .common import read,write,digest,Blocked
from .qwen_independent_probe import tensor_record,prior_formula


def probe_plan(long_probes=False,stability_smoke=False):
    if stability_smoke:
        if long_probes:raise Blocked('Select reduced stability or full probes, not both')
        return 4096,{0,1}|set(range(4080,4096))
    if not long_probes:return 1025,set((0,1,127,128,511,512,1023,1024))
    starts=(1024,4096,8192,16384,32768)
    return max(starts)+16,{i for start in starts for i in range(start,start+16)}|{0,1}


def filtered_summary(logits,special_ids):
    """Identical stable top-k/p support rule as sampling.gpu_sample, T=1."""
    import torch
    if not torch.isfinite(logits).all():raise Blocked('Nonfinite diagnostic logits')
    order=torch.argsort(logits,descending=True,stable=True)[:20]
    values=logits[order].double()
    probabilities=values.softmax(-1)
    keep=probabilities.cumsum(-1)-probabilities<.95
    filtered=probabilities*keep;filtered/=filtered.sum()
    raw=logits.double().softmax(-1)
    specials={}
    for name,token in special_ids.items():
        positions=(order==token).nonzero().flatten()
        j=int(positions[0]) if len(positions) else None
        specials[name]=dict(token_id=token,raw_probability=float(raw[token]),
            top20_eligible=j is not None,nucleus_eligible=j is not None and bool(keep[j]),
            filtered_probability=float(filtered[j]) if j is not None else 0.)
    return dict(top20_ids=order.cpu().tolist(),filtered_probabilities=filtered.cpu().tolist(),specials=specials)


def load(folder,method,side,out):
    import torch
    from .qwen_smoke import activate,run
    if side=='local':
        model,processor,_=run(folder,method,out/'loading.json','qwen8',load_only=True)
        return model,processor
    activate(folder)
    from visual_token_pruning import VTRConfig
    from visual_token_pruning.model.vtr_qwen3_vl import VTRQwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    cfg=VTRConfig(enabled=method!='Full',strategy='priortr',prune_layer=3,
        keep_ratio={'Full':1.,'PriorTR50':.5,'PriorTR25':.25}[method],keep_tokens=None,
        query_aggregation='question',head_aggregation='mean')
    lock=read(folder/'manifests/model_weights_lock.json')['qwen8']
    model=VTRQwen3VLForConditionalGeneration.from_pretrained_vtr(lock['path'],vtr_config=cfg,
        dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',local_files_only=True)
    processor=Qwen3VLProcessor.from_pretrained(lock['path'],local_files_only=True,
        min_pixels=262144,max_pixels=4194304)
    return model.eval().requires_grad_(False),processor


def run(folder,candidate,method,side,out,long_probes=False,stability_smoke=False):
    import torch
    from PIL import Image
    from .audit_qwen_inputs import question_text
    from .correctness_probe import cache_lengths
    if out.exists():raise Blocked('Immutable diagnostic attempt exists')
    out.mkdir(parents=True)
    model,processor=load(folder,method,side,out)
    from transformers import LogitsProcessorList
    row=read(candidate)
    if digest(Path(row['image_path']).read_bytes())!=row['image_sha256']:raise Blocked('Image changed')
    protocol=read(Path(__file__).resolve().parent.parent/'round3_spec.json')
    with Image.open(row['image_path']) as im:image=im.convert('RGB')
    text=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},
        {'type':'text','text':question_text(row,protocol['prompt']['suffix'])}]}],tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],images=[image],return_tensors='pt')
    inputs={k:v.to(model.device) for k,v in inputs.items()}
    total,probes=probe_plan(long_probes,stability_smoke)
    prompt_len=int(inputs['input_ids'].shape[1])
    if prompt_len>8192 or prompt_len+total>65536:raise Blocked('Context limits exceeded')
    neutral=processor.tokenizer.encode(' The image contains information that can be examined carefully.',add_special_tokens=False)
    if not neutral:raise Blocked('Empty forced prefix')
    forced=(neutral*(total//len(neutral)+1))[:total]
    specials={}
    for word in ('<think>','</think>'):
        ids=processor.tokenizer.encode(word,add_special_tokens=False)
        if len(ids)!=1:raise Blocked('Expected single-token think delimiter')
        specials[word]=ids[0]
    eos=model.generation_config.eos_token_id
    for i,token in enumerate([] if eos is None else [eos] if isinstance(eos,int) else eos):specials['EOS_'+str(i)]=token
    lock=read(folder/'manifests/model_weights_lock.json')['qwen8']
    identity=dict(side=side,method=method,model_revision=lock['revision'],
        author_revision=read(folder/'manifests/source_lock.json')['PriorTR']['revision'],uid=row['uid'],
        image_sha256=row['image_sha256'],inputs=tensor_record(inputs),forced_sha256=digest(str(forced)),
        total_steps=total,probe_steps=sorted(probes),special_ids=specials,long_probes=long_probes,
        stability_smoke=stability_smoke)
    write(out/'identity.json',identity)
    step=[0];last=[None];kv_probes={};positions=[];scores={};summaries={};selection=[];handles=[]
    original_score=original_select=None
    if method!='Full':
        strategy=model.vtr_strategy
        original_score,original_select=strategy.compute_scores,strategy.select_tokens
        def score(attention,image_token_range,config,*args,**kwargs):
            if step[0]!=0:raise Blocked('Unexpected repeated visual pruning')
            value=original_score(attention,image_token_range,config,*args,**kwargs)
            p,q,formula=prior_formula(attention,image_token_range)
            if not torch.equal(value,formula.to(value.dtype)):raise Blocked('Q/P differs')
            selection.append(dict(kind='scores',P=tensor_record(p),Q=tensor_record(q),scores=tensor_record(value)))
            return value
        def select(value,num_tokens,config,*args,**kwargs):
            answer=original_select(value,num_tokens,config,*args,**kwargs)
            selection.append(dict(kind='actual_indices',indices=answer.cpu().tolist()))
            return answer
        strategy.compute_scores=score;strategy.select_tokens=select
    def after(module,args,result):
        lengths=cache_lengths(result.past_key_values)
        if len(lengths)!=36:raise Blocked('Missing cache layer')
        if last[0] is not None and any(b!=a+1 for a,b in zip(last[0],lengths)):
            raise Blocked('Non-unit sequential cache growth')
        last[0]=lengths
        if step[0] in probes:kv_probes[str(step[0])]=lengths
    handles.append(model.register_forward_hook(after))
    def rope(module,args,kwargs):
        if step[0] in probes:
            value=args[1] if len(args)>1 else kwargs.get('position_ids')
            positions.append(dict(step=step[0],position_ids=tensor_record(value)))
    handles.append(model.model.language_model.rotary_emb.register_forward_pre_hook(rope,with_kwargs=True))
    class Force:
        def __call__(self,ids,logits):
            i=step[0]
            if stability_smoke and not torch.isfinite(logits).all():raise Blocked('Nonfinite stability logits')
            if i in probes:
                scores[i]=logits[0].detach().float().cpu()
                summaries[str(i)]=filtered_summary(logits[0],specials)
            value=torch.full_like(logits,-float('inf'));value[:,forced[i]]=0
            step[0]+=1
            if step[0]%128==0:write(out/'status.json',dict(stage='teacher_forcing',completed=step[0],
                target=total,updated_at=time.time(),native_gate_passed=False))
            return value
    started=time.time()
    try:
        generation=copy.deepcopy(model.generation_config)
        generation.eos_token_id=None;generation.forced_eos_token_id=None
        with torch.inference_mode():
            model.generate(**inputs,generation_config=generation,max_new_tokens=total,do_sample=False,
                use_cache=True,logits_processor=LogitsProcessorList([Force()]))
        if step[0]!=total or set(scores)!=probes:raise Blocked('Incomplete trajectory/probes')
        if method!='Full' and len(selection)!=2:raise Blocked('Expected exactly one visual selection')
        write(out/'status.json',dict(status='PASS',completed=step[0],wall_seconds=time.time()-started,
            native_gate_passed=False,scope='fixed teacher prefix; not natural generation or complete backend certification'))
    except BaseException as exc:
        write(out/'status.json',dict(status='FAIL',completed=step[0],error=str(exc)[:1200],native_gate_passed=False));raise
    finally:
        for handle in handles:handle.remove()
        if original_score is not None:
            strategy.compute_scores=original_score;strategy.select_tokens=original_select
        torch.save(scores,out/'logits.pt')
        write(out/'evidence.json',dict(kv_probes=kv_probes,last_layer_kv=last[0],positions=positions,
            selection=selection,filtered_probabilities=summaries,all_sequential_kv_steps_checked=step[0]))


def compare(author,local):
    import torch
    a,b=read(author/'identity.json'),read(local/'identity.json')
    if a.pop('side')!='author' or b.pop('side')!='local' or a!=b:raise Blocked('Different identities/entrypoints')
    for path in (author,local):
        if read(path/'status.json').get('status')!='PASS':raise Blocked('Missing successful full trajectory')
    x=torch.load(author/'logits.pt',map_location='cpu',weights_only=True)
    y=torch.load(local/'logits.pt',map_location='cpu',weights_only=True)
    if set(x)!=set(a['probe_steps']) or set(x)!=set(y):raise Blocked('Incomplete probe sets')
    metrics=[]
    for i in sorted(x):
        p=x[i].double().log_softmax(-1);q=y[i].double().log_softmax(-1)
        metrics.append(dict(step=i,kl=float((p.exp()*(p-q)).sum()),tv=float((p.exp()-q.exp()).abs().sum()/2)))
    ea,eb=read(author/'evidence.json'),read(local/'evidence.json')
    geometry=all(ea[k]==eb[k] for k in ('kv_probes','last_layer_kv','positions','selection','all_sequential_kv_steps_checked'))
    return dict(status='PASS' if geometry and all(m['kl']<.02 and m['tv']<.08 for m in metrics) else 'FAIL',
        metrics=metrics,geometry_exact=geometry,native_gate_passed=False,
        scope='same fixed teacher prefix, distinct loading entrypoints, shared author forward; no natural generation certification',
        probability_evidence={'author':str(author/'evidence.json'),'local':str(local/'evidence.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder');p.add_argument('--candidate');p.add_argument('--out',required=True)
    p.add_argument('--method',choices=['Full','PriorTR50','PriorTR25']);p.add_argument('--side',choices=['author','local'])
    p.add_argument('--long-probes',action='store_true');p.add_argument('--stability-smoke',action='store_true');p.add_argument('--compare',nargs=2)
    a=p.parse_args()
    if Path(a.out).exists():raise Blocked('Refuse overwrite')
    if a.compare:write(a.out,compare(*(Path(x) for x in a.compare)))
    else:
        if not all((a.folder,a.candidate,a.method,a.side)):p.error('Worker arguments required')
        run(Path(a.folder),Path(a.candidate),a.method,a.side,Path(a.out),a.long_probes,a.stability_smoke)
