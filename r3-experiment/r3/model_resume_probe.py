"""Real-model prefix reconstruction diagnostic; EOS disabled only for this test.

Tests rebuilding native KV from original inputs, not serialized KV restoration.
Never emits research answers or grants a native gate.
"""
import time
from pathlib import Path
from .common import write, Blocked, code_hash
from .correctness_probe import cache_lengths


def run(model, inputs, tokenizer, out):
    import torch
    from . import reference_stream as stream
    out=Path(out)
    if out.exists():raise Blocked('Existing diagnostic attempt')
    out.mkdir(parents=True)
    job={'job_key':'real-model-reconstruction','seed':20261022,
         'generation':dict(max_new_tokens=384,temperature=1.,top_k=20,top_p=.95)}
    original_class=stream.RequestSampler
    baseline=[];observed=[];phase=['baseline'];comparisons=[]
    class Interrupted(Exception):pass
    stop=[None]
    class Sampler(original_class):
        def __call__(self, ids, scores):
            if self.step==stop[0]:
                self.flush()
                raise Interrupted()
            return super().__call__(ids,scores)
    def capture(module,args,output):
        lengths=cache_lengths(output.past_key_values)
        index=len(observed)
        logits=output.logits[:,-1,:].detach().float().cpu()
        observed.append(lengths)
        if phase[0]=='baseline':baseline.append((lengths,logits))
        else:
            expected,reference=baseline[index]
            if lengths!=expected:raise Blocked('Reconstructed per-layer cache lengths differ')
            p=reference.softmax(-1);q=logits.softmax(-1)
            kl=float((p*(p.clamp_min(1e-30).log()-q.clamp_min(1e-30).log())).sum())
            tv=float((p-q).abs().sum()/2)
            if not torch.isfinite(logits).all() or kl>=.02 or tv>=.08:
                raise Blocked('Reconstructed logits outside fixed tolerance')
            comparisons.append({'phase':phase[0],'step':index,'kl':kl,'tv':tv,
                                'argmax_equal':bool(reference.argmax()==logits.argmax())})
    handle=model.register_forward_hook(capture)
    stream.RequestSampler=Sampler
    try:
        def execute(name,path,limit=None):
            phase[0]=name;observed.clear();stop[0]=limit
            write(out/'status.json',{'stage':name,'updated_at':time.time()})
            try:
                result=stream.generate_native(model,inputs,tokenizer,job,path,
                    extra={'eos_token_id':None,'forced_eos_token_id':None})
            except Interrupted:
                if limit is None:raise
                return None
            if limit is not None:raise Blocked('Expected diagnostic interruption did not occur')
            return result
        reference=execute('baseline',out/'baseline.prefix.json')
        checkpoint=out/'resume.prefix.json'
        execute('interrupt_at_300',checkpoint,300)
        original=checkpoint.read_bytes()
        execute('interrupt_replay_at_256',checkpoint,256)
        if checkpoint.read_bytes()!=original:raise Blocked('Interrupted replay changed original checkpoint')
        resumed=execute('reconstruct_and_continue',checkpoint)
        if resumed['generated_token_ids']!=reference['generated_token_ids']:
            raise Blocked('Reconstructed generation differs from uninterrupted baseline')
        report={'status':'PASS','native_gate_passed':False,'code_sha256':code_hash(),
                'scope':'same loaded model, original-input KV reconstruction and continuation; not fresh-process or serialized-KV restoration',
                'diagnostic_eos_disabled':True,'steps':384,'interrupted_at':300,'replay_interrupted_at':256,
                'original_checkpoint_preserved':True,'tokens_equal':True,'logit_comparisons':len(comparisons),
                'max_kl':max(c['kl'] for c in comparisons),'max_tv':max(c['tv'] for c in comparisons),
                'argmax_mismatches':sum(not c['argmax_equal'] for c in comparisons),'completed_at':time.time()}
        write(out/'report.json',report)
        write(out/'comparisons.json',comparisons)
        write(out/'status.json',{'stage':'complete',**report})
    except BaseException as exc:
        write(out/'status.json',{'stage':'failed','error_type':type(exc).__name__,'error':str(exc)[:1000]})
        raise
    finally:
        handle.remove();stream.RequestSampler=original_class
