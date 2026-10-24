"""Real four-slot paged admission, append, release and refill diagnostic."""
from pathlib import Path
import json
from .common import read,write,Blocked

def run(root,out):
    import torch
    from .native_session import NativeSession
    from .qwen_paged_decode import Backend
    from .batch_backend_probe import choose_rows
    root=Path(root);out=Path(out);out.mkdir()
    rows=choose_rows([json.loads(x) for x in (root/'manifests/frozen-inputs-v1/calibration_manifest.jsonl').read_text().splitlines()],4)
    session=NativeSession(root,'qwen8','Full',out/'loading.json');session.paged_batch_size=4
    backend=Backend(session);spec=read(Path(__file__).parent.parent/'round3_spec.json')
    def start(row):return backend.prefill(row,dict(job_key='slot-probe-'+row['uid'],generation=dict(spec['generation'],max_new_tokens=32768)))
    states=[start(row) for row in rows];initial_free=len(backend.free)
    for _ in range(4):
        logits=backend.decode(states,torch.full((4,),42,device=session.model.device,dtype=torch.long))
        assert torch.isfinite(logits).all()
    before=backend.kv_lengths(states[0]);kept=set(states[0]['pages']+states[2]['pages'])
    backend.release(states[1]);backend.release(states[3])
    states=[states[0],states[2],start(rows[1]),start(rows[3])]
    assert kept.isdisjoint(set(states[2]['pages']+states[3]['pages']))
    for _ in range(4):
        logits=backend.decode(states,torch.full((4,),43,device=session.model.device,dtype=torch.long))
        assert torch.isfinite(logits).all()
    assert backend.kv_lengths(states[0])==[n+4 for n in before]
    assert len(backend.free)==initial_free
    for s in states:backend.release(s)
    assert len(set(backend.free))==len(backend.free)==backend.pages-backend.batch_size
    write(out/'receipt.json',dict(status='PASS',slots=4,cap_reserved=32768,released_and_refilled=2,finite=True,pages_reclaimed=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),production_started=False))
