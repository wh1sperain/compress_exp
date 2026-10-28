"""Native B1 inference, immutable per-job files and recoverable original prefix/RNG.

JSONL collection is separate; never append to a possibly torn shared result file.
"""
import argparse
import importlib.metadata
import random
import re
import sys
import time
from pathlib import Path
from .common import *
from .validate_backends import admit

def adapter_row(row):
    return dict(uid=row['sample_id'],task=row['dataset'],question=row['question'],options=row['choices'],
                image_path=row['image_paths'][0],image_sha256=row['image_hash'])

def trace_fields(trace, method):
    n=trace['full_visual_tokens'];selections=trace['selections'];layers=trace['layer_count']
    if len(trace['prefill_layers'])!=layers: raise ValueError('Layer observation incomplete')
    if method!='Full' and not selections: raise ValueError('Pruning selector did not run')
    selected=selections[-1]['original_indices'] if selections else list(range(n))
    if len(set(selected))!=len(selected) or any(i<0 or i>=n for i in selected): raise ValueError('Bad visual indices')
    if method=='Full' and selections: raise ValueError('Full must bypass selection, not keep_ratio=1')
    # Pinned SparseVLM defaults merge=False; reject any unsupported merged lineage.
    params=trace.get('method_parameters',{})
    if params.get('token_merge'): raise ValueError('Merge accounting not admitted for this frozen profile')
    final_length=trace['prefill_layers'][str(layers-1)]['hidden_shape'][-2]
    if final_length != len(trace['prompt_ids'])-n+len(selected): raise ValueError('Final hidden shape/visual count mismatch')
    fused=selections[-1].get('fusion_indices',[]) if selections else []
    return dict(visual_tokens_original=n,visual_tokens_retained=len(selected),visual_tokens_removed=n-len(selected),
                keep_ratio_actual=len(selected)/n,selected_visual_token_indices=selected,
                prune_layer=[s['layer'] for s in selections],merge_enabled=False,merged_token_count=0,
                fusion_enabled=bool(params.get('hilo_use_fusion')),fusion_source_token_count=len(fused),
                fusion_source_indices=fused,method_parameters=params,image_grid_shape=trace['image_grid_thw'])

def generate(session, batch, extra, job, checkpoint):
    import torch
    import numpy as np
    from transformers import LogitsProcessorList
    sys.path.insert(0,str(ROOT.parent/'r3-experiment'))
    from r3.sampling import uniforms,gpu_sample
    from r3.prefix_checkpoint import publish
    random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
    saved=[]
    if checkpoint.exists():
        c=read(checkpoint)
        if c['run_id']!=job['run_id'] or c['job_sha256']!=digest(job): raise ValueError('Prefix binding differs')
        saved=c['generated_token_ids']
    ids=[];start=time.perf_counter();g=job['decoding_config']
    u=torch.tensor(uniforms(job['seed'],g['max_new_tokens']),device=session.model.device)
    def flush():
        publish(checkpoint,dict(run_id=job['run_id'],job_key=job['run_id'],job_sha256=digest(job),generated_token_ids=ids,
                               generated_steps=len(ids),seed=job['seed']),saved)
    def choose(input_ids,scores):
        step=len(ids)
        t=scores.argmax(dim=-1) if not g['do_sample'] else gpu_sample(scores,u[step:step+1],temperature=1.,top_k=20,top_p=.95)
        token=int(t.item())
        if step<len(saved) and saved[step]!=token: raise ValueError('Unchanged backend/RNG cannot reproduce saved prefix')
        ids.append(token)
        if len(ids)%128==0: flush()
        scores=torch.full_like(scores,-torch.inf);scores[0,token]=0.;return scores
    # Custom request sampler consumes original uniforms. HF itself follows greedy
    # forced logits in both modes, avoiding duplicate top-p/temperature processing.
    with torch.inference_mode():
        session.model.generate(**batch,**extra,do_sample=False,num_beams=1,num_return_sequences=1,
            max_new_tokens=g['max_new_tokens'],min_new_tokens=0,repetition_penalty=1.,
            temperature=None,top_p=None,top_k=None,forced_eos_token_id=None,
            logits_processor=LogitsProcessorList([choose]),use_cache=True)
    if len(ids)<len(saved): raise ValueError('Prefix shortened')
    flush();torch.cuda.synchronize()
    eos=session.model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos] if eos is not None else []
    return dict(raw_generated_token_ids=ids,raw_response=session.processor.tokenizer.decode(ids,skip_special_tokens=False),
                eos_token_ids=eos,finish_reason='eos' if ids and ids[-1] in eos else 'length' if len(ids)==32768 else 'other',
                latency_ms=(time.perf_counter()-start)*1000,replayed_prefix_tokens=len(saved))

def execute(a):
    import torch
    from r4.adapters import Session
    from r4.run import file_sha
    from .parse_outputs import parse
    spec=read(a.spec);locks=read(a.locks);ms=model_spec(spec,a.model);weights=read(a.weights_lock)
    weights=weights.get(a.model,weights)
    if not ms['model_revision'] or weights['revision']!=ms['model_revision'] or weights['model_id']!=ms['model_id']: raise ValueError('Model revision lock missing/mismatched')
    for name,entry in weights['files'].items():
        if file_sha(Path(a.checkpoint)/name)!=entry['sha256']: raise ValueError('Checkpoint hash differs')
    env=read(a.environment_lock)
    installed=importlib.metadata.version('transformers')
    if env.get('transformers_version')!=installed or not env.get('transformers_commit'): raise ValueError('Pin installed Transformers environment')
    if not re.fullmatch('[0-9a-f]{40}',env.get('code_commit','')):raise ValueError('Record full deployed Git commit')
    ident=dict(code_sha256=code_hash(),protocol_sha256=digest(spec),sources_sha256=digest(locks),weights_sha256=digest(weights),
               environment_sha256=digest(env),torch=torch.__version__,transformers=installed,gpu=torch.cuda.get_device_name(),profile='native-b1')
    formal=not a.diagnostic
    if formal:
        if not a.gate: raise ValueError('Measured admission required')
        admit(read(a.gate),ident,a.model,a.condition)
    panel={r['sample_id']:r for r in rows(a.inputs)}
    jobs=[j for j in rows(a.jobs) if j['model']==a.model and j['condition']==a.condition]
    if not jobs: raise ValueError('Empty job selection')
    if any(j['phase']=='formal' for j in jobs):
        from .plan_jobs import matrix
        if not a.decisions:raise ValueError('Formal execution needs actual precheck decision receipt')
        decisions=read(a.decisions);fallback=read(a.fallback_plan) if a.fallback_plan else None
        expected={j['run_id']:j for j in matrix(list(panel.values()),spec,'formal',decisions,fallback)}
        if any(expected.get(j['run_id'])!=j for j in jobs):raise ValueError('Formal jobs differ from frozen decisions/matrix')
    # Freeze actual source tree identity; Full uses unmodified HF class.
    s=Session(a.condition,a.checkpoint,a.author_root,ms,locks,native=a.condition=='Full',disabled=a.disabled)
    s.model.eval();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for j in jobs:
        if j['run_id']!=digest({k:v for k,v in j.items() if k!='run_id'}):raise ValueError('Job ID/content mismatch')
        if j['decoding_config']!=generation(j['decoding_mode']):raise ValueError('Changed decoder parameters')
        if a.diagnostic and j['phase']=='formal': raise ValueError('Cannot generate formal jobs under diagnostic bypass')
        r=panel[j['sample_id']]
        if j['input_sha256']!=digest(input_only(r)) or j['protocol_sha256']!=digest(spec): raise ValueError('Input/protocol changed')
        binding=dict(job=j,identity=ident,diagnostic=a.diagnostic,disabled=a.disabled)
        d=out/j['run_id'];d.mkdir(exist_ok=True)
        if (d/'binding.json').exists():
            if read(d/'binding.json')!=binding: raise ValueError('Changed resume identity')
        else: save(d/'binding.json',binding)
        if (d/'result.json').exists():
            if read(d/'result.json')['binding_sha256']!=digest(binding): raise ValueError('Corrupted success binding')
            if read(d/'success.json')['result_sha256']!=digest((d/'result.json').read_bytes()):raise ValueError('Success content changed')
            continue
        if (d/'failure.json').exists(): raise ValueError('Preserved failure requires explicit recovery; no automatic resampling')
        try:
            import numpy as np
            random.seed(j['seed']);np.random.seed(j['seed']);torch.manual_seed(j['seed']);torch.cuda.manual_seed_all(j['seed'])
            started=time.perf_counter()
            batch,extra=s.prepare(adapter_row(r),dict(uid=r['sample_id'],repeat=j['repeat']))
            raw=generate(s,batch,extra,j,d/'prefix.json')
            visual=trace_fields(s.trace,'Full' if a.disabled else j['method'])
            raw=dict(j,**raw,**visual,prompt_token_ids=s.trace['prompt_ids'],prompt_hash=s.trace['prompt_sha256'],
                     prompt=s.processor.tokenizer.decode(s.trace['prompt_ids'],skip_special_tokens=False),
                     image_hash=r['image_hash'],image_paths=r['image_paths'],image_processor_config=s.processor.image_processor.to_dict(),
                     model_revision=ms['model_revision'],code_commit=env.get('code_commit'),code_sha256=code_hash(),
                     tokenizer_sha256=digest(s.processor.tokenizer.backend_tokenizer.to_str()),
                     template_sha256=digest(s.processor.chat_template),run_status='success',binding_sha256=digest(binding),
                     trace_sha256=digest(s.trace),end_to_end_latency_ms=(time.perf_counter()-started)*1000,
                     predicted_answer=None,correct=None,diagnostic=a.diagnostic,disabled=a.disabled)
            raw['visual_budget_warning']=abs(raw['keep_ratio_actual']-j['keep_ratio_target'])>max(.01,1/raw['visual_tokens_original']) if not a.disabled else False
            save(d/'trace.json',s.trace);save(d/'result.json',parse(raw,s.processor.tokenizer))
            save(d/'success.json',dict(result_sha256=digest((d/'result.json').read_bytes())))
        except BaseException as e:
            save(d/'failure.json',dict(run_id=j['run_id'],run_status='failure',exception=type(e).__name__,message=str(e)))
            raise

def collect(folder,out):
    result=[]
    for p in sorted(Path(folder).rglob('result.json')):
        r=read(p)
        if read(p.parent/'success.json')['result_sha256']!=digest(p.read_bytes()):raise ValueError('Changed success file')
        if r['binding_sha256']!=digest(read(p.parent/'binding.json')) or r['trace_sha256']!=digest(read(p.parent/'trace.json')): raise ValueError('Result integrity check failed')
        result.append(r)
    if len({r['run_id'] for r in result})!=len(result): raise ValueError('Duplicate result')
    save_rows(out,result)
    save(str(out)+'.receipt.json',dict(records=len(result),failures=[read(p) for p in Path(folder).rglob('failure.json')],sha256=digest(Path(out).read_bytes())))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collect');p.add_argument('--out',required=True)
    for k in ('inputs','jobs','model','condition','checkpoint','author-root','weights-lock','environment-lock','gate','decisions','fallback-plan'):p.add_argument('--'+k)
    p.add_argument('--spec',default='extension_protocol.json');p.add_argument('--locks',default='sources.lock.json')
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--disabled',action='store_true');a=p.parse_args()
    if a.collect:collect(a.collect,a.out)
    else:
        if a.disabled and not a.diagnostic:raise ValueError('Disabled compressed wrapper is diagnostic only')
        execute(a)
