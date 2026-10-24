"""Synthetic full-cap KV memory stress, not answers or natural certification."""
import argparse
import json
import time
from pathlib import Path
from .common import Blocked,read,write,code_hash,digest
from .batch_backend_probe import choose_rows,make_prefix


def cache_container(state,model):
    return state if model.startswith('qwen') else state['opaque']


def advance_positions(state,model,offset):
    if type(offset) is not int or offset<0:raise Blocked('Invalid synthetic offset')
    if model.startswith('qwen'):
        state['next_position_ids']=state['next_position_ids']+offset
        state['decode_steps']+=offset
    else:
        state['opaque']['next_position']+=offset
        if 'decoded_steps' in state['opaque']:state['opaque']['decoded_steps']+=offset


def reservation(states,model,offset,scratch_factor=3):
    if scratch_factor<3:raise Blocked('Synthetic stress requires at least 3x scratch reservation')
    caches=[cache_container(state,model)['cache'] for state in states]
    if not caches or len({len(c) for c in caches})!=1:raise Blocked('Mixed cache layouts')
    padded=0
    for layer in range(len(caches[0])):
        width=max(cache[layer][0].shape[-2]+offset+1 for cache in caches)
        for kind in (0,1):
            prototype=caches[0][layer][kind]
            padded+=len(states)*prototype.shape[1]*width*prototype.shape[3]*prototype.element_size()
    return dict(final_padded_kv_bytes=padded,reservation_bytes=int(padded*scratch_factor),scratch_factor=scratch_factor)


def repeat_real_kv(value,target):
    """Allocate just the destination and repeat real KV via bounded doublings."""
    import torch
    original=value.shape[-2]
    if target<original or original<1:raise Blocked('Invalid synthetic KV length')
    shape=list(value.shape);shape[-2]=target
    result=torch.empty(shape,device=value.device,dtype=value.dtype)
    result[...,:original,:].copy_(value)
    filled=original
    while filled<target:
        copied=min(filled,target-filled)
        result[...,filled:filled+copied,:].copy_(result[...,:copied,:])
        filled+=copied
    return result


def run(folder,model,condition,out,batch_size=4,scratch_factor=3):
    import torch
    from .native_session import NativeSession
    from .batch_scheduler import backend_for
    from .natural_preparation import validate_calibration
    folder=Path(folder);out=Path(out)
    if out.exists():raise Blocked('Keep prior resource probe immutable')
    out.mkdir(parents=True)
    manifest=folder/'manifests/frozen-inputs-v1/calibration_manifest.jsonl'
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    validate_calibration(rows);selected=choose_rows(rows,batch_size)
    spec=read(Path(__file__).resolve().parent.parent/'round3_spec.json')
    cap=spec['models'][model]['max_new_tokens'];offset=cap-2
    if offset<0:raise Blocked('Unexpected protocol cap')
    identity=dict(model=model,condition=condition,batch_size=batch_size,original_output_cap=cap,
        synthetic_decode_offset=offset,diagnostic_only=True,synthetic_repeated_real_kv=True,
        natural_trajectory=False,answers_generated=False,production_certified=False,
        uids=[row['uid'] for row in selected],manifest_sha256=digest(manifest.read_bytes()),code_sha256=code_hash())
    write(out/'identity.json',identity)
    states=[];backend=None
    try:
        session=NativeSession(folder,model,condition,out/'loading.json');backend=backend_for(session)
        for row in selected:
            job=dict(uid=row['uid'],job_key='memory-probe-'+row['uid'],model=model,condition=condition,
                generation=dict(spec['generation'],max_new_tokens=cap),stage='synthetic_memory_diagnostic')
            states.append(backend.prefill(row,job))
        before=[backend.kv_lengths(state) for state in states]
        budget=reservation(states,model,offset,scratch_factor)
        torch.cuda.synchronize();torch.cuda.empty_cache();free,total=torch.cuda.mem_get_info()
        baseline=torch.cuda.memory_allocated()
        available=min(free-8*1024**3,total*.90-torch.cuda.memory_reserved())
        if budget['reservation_bytes']>available:
            result=dict(identity,**budget,status='INFEASIBLE',free_before_bytes=free,total_bytes=total,
                available_conservative_bytes=int(available),reason='3x full padded KV exceeds prechecked margin; no allocation/decode attempted',
                reduced_batch_or_cap=False)
            write(out/'receipt.json',result);return result
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        with torch.inference_mode():
            for state in states:
                container=cache_container(state,model)
                expanded=[]
                for pair in container['cache']:
                    expanded.append(tuple(repeat_real_kv(value,value.shape[-2]+offset) for value in pair))
                container['cache']=expanded
                advance_positions(state,model,offset)
            token=make_prefix(session.tokenizer,session.model.generation_config.eos_token_id,2)[0]
            tokens=torch.full((batch_size,),token,device=session.model.device,dtype=torch.long)
            logits=backend.decode(states,tokens)
            finite=bool(torch.isfinite(logits).all())
        torch.cuda.synchronize();wall=time.monotonic()-started
        after=[backend.kv_lengths(state) for state in states]
        exact=all(after[i]==[length+cap-1 for length in before[i]] for i in range(batch_size))
        peak=torch.cuda.max_memory_allocated();free_after,_=torch.cuda.mem_get_info()
        # Account for driver/other allocations seen before stress as well as
        # this process's maximum allocated tensors, not just final free memory.
        estimated_peak_free=free-max(0,peak-baseline)
        margin=min(free_after,estimated_peak_free)
        passed=finite and exact and margin>=8*1024**3
        result=dict(identity,**budget,status='PASS' if passed else 'FAIL',finite_logits=finite,
            expected_final_kv_exact=exact,prefill_layer_kv=before,final_layer_kv=after,
            peak_allocated_bytes=peak,baseline_allocated_bytes=baseline,free_after_bytes=free_after,
            conservative_peak_free_bytes=margin,required_free_margin_bytes=8*1024**3,
            wall_seconds=wall,reduced_batch_or_cap=False,
            scope='one synthetic full-cap memory/finite step only; repeated KV is not valid natural attention evidence')
        write(out/'receipt.json',result);return result
    except BaseException as exc:
        write(out/'failure.json',dict(identity,status='FAIL',error_type=type(exc).__name__,
            error=str(exc)[:1200],automatic_retry=False,reduced_batch_or_cap=False))
        raise
    finally:
        if backend:
            for state in states:backend.release(state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--model',required=True)
    p.add_argument('--condition',required=True);p.add_argument('--out',required=True)
    p.add_argument('--batch-size',type=int,choices=[2,3,4],default=4);p.add_argument('--scratch-factor',type=float,default=3)
    a=p.parse_args();result=run(a.folder,a.model,a.condition,a.out,a.batch_size,a.scratch_factor)
    print(json.dumps({k:result[k] for k in ['status','model','condition','batch_size','original_output_cap','diagnostic_only']}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
