"""Bounded raw-logit diagnostics for new batched backends; never research jobs."""
import argparse
import copy
import importlib
import json
import time
from pathlib import Path
from .common import Blocked, digest, read, write, code_hash
from .manifest import TASKS
from .correctness_probe import cache_lengths


def choose_rows(rows, count):
    if count < 1 or count > len(rows):
        raise Blocked('Invalid diagnostic batch size')
    groups = {task: sorted([r for r in rows if r['task'] == task], key=lambda r:r['uid']) for task in TASKS}
    selected, images = [], set()
    for offset in range(max(map(len, groups.values()))):
        for task in TASKS:
            if offset >= len(groups[task]):continue
            row = groups[task][offset]
            if row['image_sha256'] in images:continue
            selected.append(row);images.add(row['image_sha256'])
            if len(selected) == count:return selected
    raise Blocked('Insufficient unique calibration images')


def make_prefix(tokenizer, eos, steps):
    if steps < 2:raise Blocked('Need at least prefill and one decode step')
    eos = set([] if eos is None else [eos] if isinstance(eos,int) else eos)
    base = [n for n in tokenizer.encode(' The image contains information that can be examined carefully.', add_special_tokens=False) if n not in eos]
    if not base:raise Blocked('Empty non-EOS fixed prefix')
    return (base*(steps//len(base)+1))[:steps]


def probe_steps(steps):
    return set(range(steps)) if steps<=32 else {i for i in (0,1,31,127,1023,4095,steps-1) if i<steps}


def native_trace(session, row, prefix):
    import torch
    from transformers import LogitsProcessorList
    inputs, metadata = session.prepare(row)
    logits, lengths = {}, []
    probes=probe_steps(len(prefix))
    def capture(module,args,output):
        raw=output.logits[0,-1,:].detach()
        if not torch.isfinite(raw).all():raise Blocked('Non-finite native logits')
        if len(lengths) in probes:logits[len(lengths)]=raw.float().cpu()
        lengths.append(cache_lengths(output.past_key_values))
    class Force:
        def __call__(self,ids,scores):
            index = len(lengths)-1
            forced = torch.full_like(scores,-float('inf'))
            forced[:,prefix[index]] = 0.
            return forced
    config = copy.deepcopy(session.model.generation_config)
    config.do_sample = False;config.num_beams = 1;config.max_new_tokens = len(prefix)
    config.min_new_tokens = 0;config.eos_token_id = None;config.forced_eos_token_id = None
    config.use_cache = True
    hook = session.model.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            session.model.generate(**inputs,generation_config=config,logits_processor=LogitsProcessorList([Force()]))
    finally:hook.remove()
    if len(lengths) != len(prefix):raise Blocked('Native teacher forcing ended early')
    for before,after in zip(lengths,lengths[1:]):
        if len(before)!=len(after) or any(b!=a+1 for a,b in zip(before,after)):
            raise Blocked('Native KV did not grow by one')
    return dict(logits=logits,kv=lengths,metadata=metadata)


def distribution_metrics(reference,candidate):
    import torch
    if reference.shape!=candidate.shape or not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise Blocked('Invalid raw logits')
    lp=torch.log_softmax(reference.double(),dim=-1);lq=torch.log_softmax(candidate.double(),dim=-1)
    p=lp.exp();q=lq.exp()
    return dict(kl_reference_candidate=float((p*(lp-lq)).sum()),
                total_variation=float((p-q).abs().sum()/2),
                argmax_mismatch=bool(reference.argmax()!=candidate.argmax()),
                max_abs_logit_difference=float((reference-candidate).abs().max()))


def run(folder,model_key,condition,out,steps=32,batch_size=4,benchmark=False):
    import torch
    from .native_session import NativeSession
    from .natural_preparation import validate_calibration
    folder=Path(folder);out=Path(out)
    if out.exists():raise Blocked('Diagnostic attempt already exists')
    out.mkdir(parents=True)
    manifest=folder/'manifests/frozen-inputs-v1/calibration_manifest.jsonl'
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    if model_key=='qwen4':
        from .supplemental_inputs import attach
        extra=read(folder/'telemetry/qwen4_supplemental_audit_launch.json')
        rows,_=attach(rows,extra['out'],extra['status_path'])
    validate_calibration(rows)
    selected=choose_rows(rows,batch_size)
    spec=read(Path(__file__).resolve().parent.parent/'round3_spec.json')
    identity=dict(model=model_key,condition=condition,steps=steps,batch_size=batch_size,
        manifest_sha256=digest(manifest.read_bytes()),code_sha256=code_hash(),uids=[r['uid'] for r in selected],
        tasks=[r['task'] for r in selected],stage='short_batch_numeric_diagnostic',research_started=False,
        full_native_engine_certification=False,labels_read=False)
    write(out/'identity.json',identity)
    try:
        session=NativeSession(folder,model_key,condition,out/'loading.json')
        suffix='qwen' if model_key.startswith('qwen') else 'mini' if model_key=='mini8' else 'llava'
        prefix=make_prefix(session.tokenizer,session.model.generation_config.eos_token_id,steps)
        write(out/'forcing_prefix.json',dict(token_ids=prefix,diagnostic_only=True,original_research_cap=spec['models'][model_key]['max_new_tokens']))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        references=[]
        for row in selected:
            references.append(native_trace(session,row,prefix))
            write(out/'status.json',dict(stage='native',completed=len(references),total=batch_size))
        torch.cuda.synchronize();native_wall=time.monotonic()-started
        native_peak=torch.cuda.max_memory_allocated()
        # Build candidate only after independent reference; its pool/config must
        # not consume reference workspace or affect reference numerical kernels.
        backend=importlib.import_module('r3.'+suffix+'_batch_decode').Backend(session)
        # Keep complete raw distributions as diagnostic evidence, not answers.
        torch.save(dict(prefix=prefix,uids=identity['uids'],logits=[r['logits'] for r in references],
                        kv=[r['kv'] for r in references]),out/'native.pt')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
        states=[]
        for row in selected:
            job=dict(model=model_key,condition=condition,uid=row['uid'],job_key='diagnostic-'+row['uid'],
                     seed=0,stage='short_batch_numeric_diagnostic',generation=dict(spec['generation'],max_new_tokens=spec['models'][model_key]['max_new_tokens']))
            states.append(backend.prefill(row,job))
        probes=probe_steps(steps)
        candidate=[{0:state['logits'][0].detach().float().cpu()} for state in states]
        kv=[[backend.kv_lengths(state)] for state in states]
        for step in range(1,steps):
            tokens=torch.full((batch_size,),prefix[step-1],device=session.model.device,dtype=torch.long)
            logits=backend.decode(states,tokens)
            if not torch.isfinite(logits).all():raise Blocked('Non-finite batch logits')
            for i,state in enumerate(states):
                if step in probes:candidate[i][step]=logits[i].detach().float().cpu()
                kv[i].append(backend.kv_lengths(state))
            if step%8==0:write(out/'status.json',dict(stage='batch',completed=step+1,total=steps))
        torch.cuda.synchronize();batch_wall=time.monotonic()-started
        batch_peak=torch.cuda.max_memory_allocated()
        torch.save(dict(prefix=prefix,uids=identity['uids'],logits=candidate,kv=kv),out/'batch.pt')
        metrics=[];passed=True
        for i,row in enumerate(selected):
            for step in sorted(probes):
                item=distribution_metrics(references[i]['logits'][step],candidate[i][step])
                item.update(uid=row['uid'],step=step,kv_exact=kv[i][step]==references[i]['kv'][step])
                item['passed']=item['kl_reference_candidate']<.02 and item['total_variation']<.08 and item['kv_exact']
                passed=passed and item['passed'];metrics.append(item)
        all_kv_exact=all(kv[i]==references[i]['kv'] for i in range(batch_size))
        passed=passed and all_kv_exact
        write(out/'per_step.json',metrics)
        receipt=dict(identity,status='PASS' if passed else 'FAIL',backend=getattr(backend,'name',type(backend).__module__),
            scope='short teacher-forced batch numeric only; not full long-prefix/natural certification',
            kl_max=max(m['kl_reference_candidate'] for m in metrics),tv_max=max(m['total_variation'] for m in metrics),
            argmax_mismatch_steps=sum(m['argmax_mismatch'] for m in metrics),kv_exact=all_kv_exact,
            compared_logit_steps=sorted(probes),finite_and_kv_checked_steps=steps,
            native_serial_wall_seconds=native_wall,batch_wall_seconds=batch_wall,
            timing_scope='full prefill plus short decode including diagnostic CPU logit transfers; excludes model loading',
            native_peak_allocated_bytes=native_peak,batch_peak_allocated_bytes=batch_peak,
            final_persistent_kv_bytes=sum(backend.kv_bytes(state) for state in states),
            position_validation_scope='position effects covered by fixed-prefix logits and all-layer KV; not a standalone raw position trace',
            production_certified=False,synthetic_fixed_prefix=True)
        write(out/'receipt.json',receipt)
        if benchmark and passed:
            from .paged_performance_probe import run as measure
            measure(backend,states,out/'performance.json')
        return receipt
    except BaseException as exc:
        write(out/'failure.json',dict(identity,status='FAIL',error_type=type(exc).__name__,error=str(exc)[:1500],production_certified=False))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--model',required=True,choices=['llava7','mini8','qwen8','qwen4']);p.add_argument('--condition',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=32);p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args();result=run(a.folder,a.model,a.condition,a.out,a.steps,a.batch_size)
    print(json.dumps({k:result[k] for k in ['status','model','condition','batch_size','steps','kl_max','tv_max','kv_exact','native_serial_wall_seconds','batch_wall_seconds']}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
