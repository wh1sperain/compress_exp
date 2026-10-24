"""Bounded concurrency sweep on calibration images; no research generation."""
import argparse,json,time,statistics,math
from pathlib import Path
from .common import read,write,Blocked

def run(root,out,batch):
    import torch
    from .native_session import NativeSession
    from .qwen_paged_decode import Backend,Native
    from .batch_backend_probe import choose_rows,distribution_metrics
    from .sampling import gpu_sample
    root=Path(root);out=Path(out);out.mkdir()
    try:
        rows=choose_rows([json.loads(x) for x in (root/'manifests/frozen-inputs-v1/calibration_manifest.jsonl').read_text().splitlines()],batch)
        session=NativeSession(root,'qwen8','Full',out/'loading.json')
        session.paged_batch_size=batch;session.paged_cache_gib=batch*6
        engine=Backend(session);dense=Native(session)
        spec=read(Path(__file__).parent.parent/'round3_spec.json')
        states=[];torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for row in rows:
            states.append(engine.prefill(row,dict(job_key='concurrency-'+row['uid'],generation=dict(spec['generation'],max_new_tokens=32768))))
        torch.cuda.synchronize();prefill=time.perf_counter()-start
        total=torch.cuda.mem_get_info()[1];peak=torch.cuda.max_memory_allocated()
        if total-peak<8*1024**3:raise Blocked('Less than 8GiB peak workspace margin')
        # Independent dense decode on exact same short caches for the new batch.
        copies=[]
        for s in states:
            c={k:(v.clone() if torch.is_tensor(v) else v) for k,v in s.items() if k not in ('pages','cache')}
            pairs=[]
            for layer in range(len(engine.layers)):
                pair=[]
                for kind in range(2):
                    v=torch.cat([engine.kv[layer,p,kind] for p in s['pages'][:math.ceil(s['length']/256)]])[:s['length']]
                    pair.append(v.transpose(0,1).unsqueeze(0).contiguous())
                pairs.append(tuple(pair))
            c.update(owner=id(dense),cache=pairs);copies.append(c)
        tokens=torch.full((batch,),42,device=session.model.device,dtype=torch.long)
        with torch.inference_mode():
            a=dense.decode(copies,tokens);b=engine.decode(states,tokens)
        metrics=[distribution_metrics(a[i],b[i]) for i in range(batch)]
        if max(m['kl_reference_candidate'] for m in metrics)>=.02 or max(m['total_variation'] for m in metrics)>=.08:raise Blocked('Batch numeric tolerance failed')
        del copies,pairs,pair,c,a,b,v;torch.cuda.empty_cache()
        scenarios={};u=torch.linspace(.1,.9,batch,device=session.model.device,dtype=torch.float64)
        for name in ['short','mixed','long']:
            # Synthetic cache lengths are equal-distribution workload probes,
            # not natural answers or regenerated research records.
            targets=[2048 if name=='short' else 30000 if name=='long' or i%2==0 else 2048 for i in range(batch)]
            with torch.inference_mode():
                for s,n in zip(states,targets):
                    for layer in range(len(engine.layers)):
                        for kind in range(2):
                            seed=engine.kv[layer,s['pages'][0],kind].clone()
                            for j in range(math.ceil(n/256)):engine.kv[layer,s['pages'][j],kind].copy_(seed)
                    s['length']=n;s['decode_steps']=0;s['next_position_ids'].fill_(n)
                for _ in range(3):engine.decode(states,tokens)
                samples=[]
                for _ in range(3):
                    torch.cuda.synchronize();start=time.perf_counter()
                    for step in range(64):
                        logits=engine.decode(states,tokens)
                        sampled=gpu_sample(logits,u,temperature=1.,top_k=20,top_p=.95)
                        # Include the production host transfer used for EOS handling.
                        sampled.tolist()
                    torch.cuda.synchronize();samples.append(time.perf_counter()-start)
                scenarios[name]=dict(tokens_per_second=64*batch/statistics.median(samples),seconds=samples)
        # Release/refill exercises physical page reuse at this exact slot count.
        for s in states[::2]:engine.release(s)
        for i in range(0,batch,2):
            states[i]=engine.prefill(rows[i],dict(job_key='refill-'+rows[i]['uid'],generation=dict(spec['generation'],max_new_tokens=32768)))
        logits=engine.decode(states,tokens)
        assert torch.isfinite(logits).all()
        for s in states:engine.release(s)
        assert len(set(engine.free))==len(engine.free)==engine.pages-batch
        peak=torch.cuda.max_memory_allocated()
        if total-peak<8*1024**3:raise Blocked('Insufficient peak margin including refill')
        result=dict(status='PASS',batch=batch,cache_gib=batch*6,prefill_seconds=prefill,prefill_per_image=prefill/batch,
            scenarios=scenarios,peak_allocated_bytes=peak,total_bytes=total,margin_bytes=total-peak,
            kl_max=max(m['kl_reference_candidate'] for m in metrics),tv_max=max(m['total_variation'] for m in metrics),
            scope='calibration prefill plus synthetic cache decode and production GPU sampler; not natural end-to-end throughput',production_started=False)
        write(out/'receipt.json',result);return result
    except BaseException as e:
        write(out/'failure.json',dict(status='FAIL',batch=batch,error_type=type(e).__name__,error=str(e)[:1500],production_started=False));raise

if __name__=='__main__':
    import sys
    sys.path.append('/root/autodl-tmp/Compress/.venv-vllm/lib/python3.12/site-packages')
    p=argparse.ArgumentParser();p.add_argument('--folder');p.add_argument('--out');p.add_argument('--batch',type=int);a=p.parse_args()
    run(a.folder,a.out,a.batch)
