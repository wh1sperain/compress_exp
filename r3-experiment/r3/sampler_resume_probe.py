"""Synthetic GPU sampler/replay regression; does not test model KV restoration."""
import argparse,time
from pathlib import Path
from .common import read,write,Blocked,code_hash


def run(out):
    import torch
    from .reference_stream import RequestSampler
    from .sampling import cpu_sample,uniforms
    out=Path(out)
    if out.exists():raise Blocked('Diagnostic output already exists')
    out.mkdir(parents=True)
    if not torch.cuda.is_available():raise Blocked('GPU required')
    job={'job_key':'synthetic-gpu-resume','seed':20261022,
         'generation':dict(max_new_tokens=384,temperature=1.,top_k=20,top_p=.95)}
    path=out/'prefix.json';dummy=torch.zeros((1,1),device='cuda',dtype=torch.long)
    # BF16 logits fixed before each replay; no model or dataset is loaded.
    source=torch.sin(torch.arange(384*128,dtype=torch.float32).reshape(384,128)*.07).bfloat16()
    source[0].zero_()
    scores=source.cuda();u=uniforms(job['seed'],384)
    expected=[cpu_sample(source[i].float().numpy(),u[i]) for i in range(384)]
    first=RequestSampler(job,path,'cuda')
    for i in range(300):first(dummy,scores[i:i+1])
    assert first.flush()==expected[:300]
    original=path.read_bytes()
    interrupted=RequestSampler(job,path,'cuda')
    for i in range(256):interrupted(dummy,scores[i:i+1])
    assert path.read_bytes()==original
    assert interrupted.flush()==expected[:256]
    assert path.read_bytes()==original
    mismatch=RequestSampler(job,path,'cuda')
    for i in range(100):mismatch(dummy,scores[i:i+1])
    wrong=torch.full_like(scores[100:101],-100.)
    wrong[0,(expected[100]+1)%128]=100.
    try:mismatch(dummy,wrong)
    except Blocked:pass
    else:raise AssertionError('Changed replay token was accepted')
    assert path.read_bytes()==original
    resumed=RequestSampler(job,path,'cuda')
    for i in range(384):resumed(dummy,scores[i:i+1])
    assert resumed.flush()==expected
    assert read(path)['generated_steps']==384
    assert any(p.read_bytes()==original for p in path.with_name(path.name+'.history').glob('*.json'))
    report={'status':'PASS','scope':'synthetic GPU sampler and token/RNG replay only; no model cache restoration',
            'native_gate_passed':False,'code_sha256':code_hash(),'cpu_gpu_tokens_equal':384,
            'original_prefix':300,'interrupted_replay':256,'resumed_prefix':384,
            'mismatch_rejected':True,'original_preserved':True,'completed_at':time.time()}
    write(out/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
