"""GPU algorithm-component probe. Never emits a full native/engine gate."""
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace as NS
from .common import read, write, code_hash, Blocked
from .native_sources import AuthorSources, visionzip_arguments


def probe(lock_path):
    import torch
    import numpy as np
    from .sampling import uniforms, cpu_sample, gpu_sample
    if not torch.cuda.is_available(): raise Blocked('CUDA required')
    sources = AuthorSources(lock_path)
    torch.manual_seed(20261022)
    device = 'cuda'; checks = []
    attn = torch.rand(1, 4, 600, 600, device=device, dtype=torch.bfloat16)
    attn = torch.softmax(attn.float().masked_fill(torch.ones(600,600,device=device,dtype=torch.bool).triu(1), -float('inf')), -1).bfloat16()
    config = NS(query_aggregation='question', head_aggregation='mean', debug=False)
    for backbone in ('llava', 'qwen3'):
        fn = sources.priortr(backbone)
        score = fn(None, attn, (5,581), config)
        assert score.shape == (576,) and torch.isfinite(score).all()
        for k in (144,288):
            selected = score.topk(k).indices
            assert len(selected.unique()) == k
        checks.append({'method':'PriorTR','backbone':backbone,'query_indices':[581,600 if backbone=='llava' else 599], 'finite':True})
    agile = sources.agile()
    features = torch.randn(576, 128, device=device, dtype=torch.bfloat16)
    rank = agile['effective_rank'](features).item()
    norms = features/(features.norm(dim=-1,keepdim=True)+1e-8)
    distance = 1-norms@norms.T
    attention = torch.rand(1,576,device=device,dtype=torch.bfloat16)
    for k in (144,288):
        selected = agile['select_diverse_tokens_by_attention_and_distance'](attention,distance,rank,max_tokens=k)
        assert len(selected.unique()) == k
        checks.append({'method':'AgilePruner','keep':k,'unique':k,'erank_reference':agile['ERANK_AVG_REF'],'tau_max':agile['TAU_MAX']})
    # Execute the actual author forward with synthetic penultimate activations;
    # this checks its tensor branch and public-API budget translation, not CLIP.
    hidden = torch.randn(1,577,32,device=device,dtype=torch.bfloat16)
    metric = torch.randn(1,577,8,device=device,dtype=torch.bfloat16)
    weights = torch.rand(1,2,577,577,device=device,dtype=torch.bfloat16)
    for k in (144,288):
        args = visionzip_arguments(k)
        class Tower:
            _info={'dominant':args['dominant']-1,'contextual':args['contextual']}
            vision_model=NS(encoder=NS(layers=[NS(metric=metric),NS()]))
            def __call__(self,*a,**kw):return NS(attentions=[weights,weights],hidden_states=[hidden,hidden])
        fake = NS(device=device,dtype=torch.bfloat16,vision_tower=Tower())
        output,indices=sources.visionzip()(fake,torch.empty(1,3,336,336,device=device,dtype=torch.bfloat16))
        assert output.shape == (1,k,32) and torch.isfinite(output).all()
        assert indices[0,0].item()==0
        checks.append({'method':'VisionZip','public_arguments':args,'actual_tokens':output.shape[1],'cls_retained':True})
    logits = torch.randn(64,200,device=device,dtype=torch.bfloat16)
    logits[0].zero_()
    u = uniforms(1234,64)
    for t in (1.,.6):
        actual=gpu_sample(logits,torch.tensor(u,device=device),temperature=t).cpu().tolist()
        reference=[cpu_sample(row,x,temperature=t) for row,x in zip(logits.float().cpu().numpy(),u)]
        assert actual==reference
    checks.append({'component':'sampler','requests':64,'temperatures':[1.,.6],'CPU_GPU_token_match':True})
    return {'component_probe_passed':True,'native_gate_passed':False,'production_ready':False,
            'scope':'synthetic algorithm components only; no model weights, Full/logits/cache/long/natural equivalence tested',
            'code_sha256':code_hash(),'torch':torch.__version__,'device':torch.cuda.get_device_name(),'checks':checks}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-lock',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();write(a.out,probe(a.source_lock))


if __name__=='__main__':main()
