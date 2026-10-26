"""Execute one explicitly registered, human-grounded mechanism branch.

No sample selection or annotations are invented here. Parent success stays immutable.
"""
import argparse
from pathlib import Path
from .core import *
from .run import identity,file_sha
from .gates import admit,validate_trace
from .interventions import rescue_pair,map_region,verified_fact_pair,prefix_triplet,stop_prefix
from .continuation import generate

KINDS=('visual_rescue','random_rescue','fact','neutral_text','original_prefix','corrected_prefix','neutral_edit','stop')

def resolve(recipe,parent,trace,session,row):
    kind=recipe['kind']
    if kind not in KINDS:raise ValueError('Unregistered mechanism')
    if kind=='stop' and session.name not in ('Full','PriorTR25','FastV25'):raise ValueError('Unregistered stop condition')
    if kind in ('fact','neutral_text','original_prefix','corrected_prefix','neutral_edit') and session.name!='PriorTR25':
        raise ValueError('Unregistered mechanism condition')
    if sha(parent['generated_token_ids'])!=recipe['parent_output_sha256']:raise ValueError('Parent changed')
    if kind!='stop' and (len(set(recipe.get('approval',{}).get('reviewers',[])))<2 or not recipe['approval'].get('adjudicated')):
        raise ValueError('Human fact/first-error adjudication absent')
    override=None;fact=None;continuation=None;reuse=False
    if kind in ('visual_rescue','random_rescue'):
        if session.name not in ('PriorTR25','FastV25'):raise ValueError('Unregistered rescue method')
        evidence=set()
        for roi in recipe['regions']:
            evidence.update(map_region(roi,recipe['token_boxes'],row['image_sha256'],recipe['mapping_receipt']))
        selected=trace['selections'][-1]['original_indices']
        pair=rescue_pair(selected,sorted(evidence),trace['full_visual_tokens'],recipe['pair_key'])
        override=pair['rescue' if kind=='visual_rescue' else 'placebo']
    elif kind in ('fact','neutral_text'):
        pair=verified_fact_pair(recipe['fact'],recipe['neutral'],session.processor.tokenizer,recipe['approval'])
        fact=pair['fact' if kind=='fact' else 'neutral']
    elif kind=='stop':
        continuation=stop_prefix(parent['generated_token_ids'],recipe['budget'],
            think_close=session.processor.tokenizer.convert_tokens_to_ids('</think>'),
            eos_ids=([parent['generation_eos_token_id']] if isinstance(parent['generation_eos_token_id'],int) else parent['generation_eos_token_id']),
            structural_ids=tuple(session.processor.tokenizer.all_special_ids))
        reuse=continuation.get('reuse_natural',False)
    else:
        candidates=prefix_triplet(parent['generated_token_ids'],recipe['error_start_token'],recipe['error_end_token'],
            recipe['corrected_token_ids'],recipe['neutral_token_ids'])
        continuation=candidates[{'original_prefix':'original','corrected_prefix':'corrected','neutral_edit':'neutral'}[kind]]
    return override,fact,continuation,reuse

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('recipe','parent','row','checkpoint','weights-lock','author-root','gate','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--spec',default='protocol.json');p.add_argument('--locks',default='sources.lock.json');a=p.parse_args()
    spec=read(a.spec);locks=read(a.locks);recipe=read(a.recipe);parent=read(a.parent);row=read(a.row)
    if not parent.get('formal'):raise ValueError('Use completed formal R4 parent only')
    trace=read(Path(a.parent).with_name('trace.json'))
    if sha(trace)!=parent['trace_sha256']:raise ValueError('Parent trace changed')
    ident=identity(spec,locks,a.checkpoint,a.weights_lock);job=dict(parent['job'])
    if job['input_sha256']!=sha(input_only(row)):raise ValueError('Wrong parent input')
    gate=read(a.gate);g=admit(gate,ident,job['condition'])
    for ev in g['evidence_files']:
        if file_sha(ev['path'])!=ev['sha256']:raise ValueError('Gate evidence changed')
    if g.get('intervention_checks',{}).get(recipe['kind'])!='PASS':raise ValueError('Intervention replay/resume not validated')
    from .adapters import Session
    session=Session(job['condition'],a.checkpoint,a.author_root,spec,locks)
    override,fact,cont,reuse=resolve(recipe,parent,trace,session,row)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    binding=dict(parent_job_key=job['job_key'],recipe=recipe,identity=ident)
    write_once(out/'binding.json',binding)
    if reuse:
        write_once(out/'reuse.json',dict(parent_job_key=job['job_key'],parent_output_sha256=recipe['parent_output_sha256'],
            reason='natural thinking ended before registered budget',new_generation=False))
    else:
        job['job_key']=sha(binding)
        try:
            expected=trace['selections'][-1]['original_indices'] if override is not None else None
            batch,extra=session.prepare(row,job,override_indices=override,extra_fact=fact,expected_original_indices=expected)
            if cont is None:
                from r3.reference_stream import generate_native
                result=generate_native(session.model,batch,session.processor.tokenizer,job,out/'prefix.json',extra=extra)
            else:result=generate(session,batch,extra,job,cont)
            validate_trace(session.trace,session.trace['full_visual_tokens'],session.method)
            write_once(out/'trace.json',session.trace)
            write_once(out/'intervention_result.json',dict(result,parent_job_key=parent['job']['job_key'],
                binding_sha256=sha(binding),trace_sha256=sha(session.trace),kind=recipe['kind'],benchmark_result=False))
        except Exception as e:
            write_once(out/'failure.json',dict(type=type(e).__name__,message=str(e)))
            raise
