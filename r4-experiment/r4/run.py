"""Immutable native reference worker. Diagnostic outputs never count as formal results."""
import argparse
import hashlib
import importlib.metadata
import sys
import time
from pathlib import Path
from .core import *
from .gates import admit,validate_trace

R3=ROOT.parent/'r3-experiment'
sys.path.insert(0,str(R3))

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def identity(spec, locks, checkpoint, weights_lock):
    import torch
    w=read(weights_lock)
    if 'qwen8' in w:w=w['qwen8']
    if w['revision']!=spec['model_revision'] or w['model_id']!=spec['model_id']:
        raise ValueError('Wrong model weights')
    for name,entry in w['files'].items():
        if file_sha(Path(checkpoint)/name)!=entry['sha256']:raise ValueError('Changed checkpoint: '+name)
    return dict(code_sha256=code_hash(),spec_sha256=sha(spec),sources_sha256=sha(locks),
        weights_sha256=sha(w['files']),model_revision=w['revision'],
        r3_dependencies_sha256=sha({p.name:file_sha(p) for p in (R3/'r3').glob('*.py')}),
        torch=torch.__version__,transformers=importlib.metadata.version('transformers'),
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),backend=spec['attention_backend'],
        execution_profile='author-native-b1-v1')

def execute(panel,jobs,name,session,out,ident,formal=False):
    from r3.reference_stream import generate_native
    rows={r['uid']:r for r in panel}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    completed=0
    for job in jobs:
        if job['condition']!=name:continue
        row=rows[job['uid']]
        if job['input_sha256']!=sha(input_only(row)):raise ValueError('Changed job input')
        if job['protocol_sha256']!=ident['spec_sha256']:raise ValueError('Changed protocol')
        d=out/job['job_key'];d.mkdir(exist_ok=True)
        binding=dict(job=job,identity=ident,formal=formal)
        if (d/'binding.json').exists():
            if read(d/'binding.json')!=binding:raise ValueError('Existing job identity changed')
        else:write_once(d/'binding.json',binding)
        if (d/'result.json').exists():
            result=read(d/'result.json')
            if result['binding_sha256']!=sha(binding):raise ValueError('Success binding mismatch')
            completed+=1;continue
        if (d/'failure.json').exists():raise ValueError('Prior failure preserved; explicit recovery release required')
        try:
            batch,extra=session.prepare(row,job)
            result=generate_native(session.model,batch,session.processor.tokenizer,job,d/'prefix.json',extra=extra)
            validate_trace(session.trace,session.trace['full_visual_tokens'],session.method)
            write_once(d/'trace.json',session.trace)
            write_once(d/'result.json',dict(result,job=job,binding_sha256=sha(binding),
                trace_sha256=sha(session.trace),formal=formal,labels_read=False,
                generation_eos_token_id=session.model.generation_config.eos_token_id))
            completed+=1
        except Exception as e:
            write_once(d/'failure.json',dict(type=type(e).__name__,message=str(e),time=time.time()))
            raise
    write_once(out/f'completion-{name}.json',dict(completed=completed,condition=name,formal=formal,identity=ident))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('panel','jobs','condition','checkpoint','author-root','weights-lock','out'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--spec',default='protocol.json');p.add_argument('--locks',default='sources.lock.json')
    p.add_argument('--gate');p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args();spec=read(a.spec);locks=read(a.locks)
    ident=identity(spec,locks,a.checkpoint,a.weights_lock)
    if not a.diagnostic:
        if not a.gate:raise SystemExit('Formal run needs measured GPU admission')
        receipt=read(a.gate);g=admit(receipt,ident,a.condition)
        for evidence in g['evidence_files']:
            if file_sha(evidence['path'])!=evidence['sha256']:raise ValueError('Gate evidence changed')
    from .adapters import Session
    session=Session(a.condition,a.checkpoint,a.author_root,spec,locks)
    execute(read(a.panel),read(a.jobs),a.condition,session,a.out,ident,formal=not a.diagnostic)
