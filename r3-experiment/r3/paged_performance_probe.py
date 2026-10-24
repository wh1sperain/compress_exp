"""Bounded synthetic long-cache comparison; never research answers."""
import math,time
from .common import write,Blocked
from .batch_backend_probe import distribution_metrics

def run(backend,states,out,steps=16):
    import torch
    from .qwen_paged_decode import Native
    dense=Native(backend.session)
    copies=[]
    # Repeat actual prefill KV to model a long active request; this explicitly
    # is not a natural trajectory and cannot certify research equivalence.
    for j,s in enumerate(states):
        n=s['length'];target=max(n,30000) if j==0 else n
        if math.ceil((target+steps+4)/backend.page_size)>len(s['pages']):
            raise Blocked('Synthetic target exceeds reserved pages')
        pairs=[]
        with torch.inference_mode():
            for layer in range(len(backend.layers)):
                values=[]
                for kind in range(2):
                    original=torch.cat([backend.kv[layer,p,kind] for p in s['pages'][:math.ceil(n/backend.page_size)]])[:n]
                    expanded=original.repeat((math.ceil(target/n),1,1))[:target].contiguous()
                    for k in range(math.ceil(target/backend.page_size)):
                        piece=expanded[k*backend.page_size:(k+1)*backend.page_size]
                        backend.kv[layer,s['pages'][k],kind,:len(piece)].copy_(piece)
                    values.append(expanded.transpose(0,1).unsqueeze(0).contiguous())
                pairs.append(tuple(values))
        s['next_position_ids']=s['next_position_ids']+(target-n)
        s['decode_steps']+=target-n;s['length']=target
        c={k:(v.clone() if torch.is_tensor(v) else v) for k,v in s.items() if k not in ('pages','cache')}
        c.update(owner=id(dense),cache=pairs);copies.append(c)
    tokens=torch.full((len(states),),42,device=backend.model.device,dtype=torch.long)
    metrics=[]
    for _ in range(2):
        a=dense.decode(copies,tokens);b=backend.decode(states,tokens)
        metrics.extend(distribution_metrics(a[i],b[i]) for i in range(len(states)))
    times={}
    for name,engine,items in [('dense',dense,copies),('paged',backend,states)]:
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(steps):engine.decode(items,tokens)
        torch.cuda.synchronize();wall=time.perf_counter()-start
        times[name]={'seconds':wall,'aggregate_tokens_per_second':steps*len(items)/wall}
    result=dict(scope='synthetic repeated KV full-model decode only; excludes prefill/sampling; not natural throughput',
        active_requests=len(states),steps=steps,lengths=[s['length'] for s in states],timing=times,
        speedup=times['dense']['seconds']/times['paged']['seconds'],
        kl_max=max(m['kl_reference_candidate'] for m in metrics),tv_max=max(m['total_variation'] for m in metrics))
    write(out,result)
    return result
