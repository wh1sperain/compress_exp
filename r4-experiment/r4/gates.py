"""Admission checks require measured evidence bound to this code, inputs and environment."""
from .core import sha, code_hash

CHECKS=('full_disabled_logits','full_keep100_logits','greedy32','long_prefix_logits',
        'text_preserved','mrope','deepstack','mask','kv36','indices_and_geometry',
        'natural_long','near_cap_cache','backend_bridge','sampler_resume')

def admit(receipt, identity, condition):
    if receipt.get('identity')!=identity or identity.get('code_sha256')!=code_hash():
        raise ValueError('Gate identity/code mismatch')
    g=receipt.get('conditions',{}).get(condition,{})
    if any(g.get('checks',{}).get(k)!='PASS' for k in CHECKS):
        raise ValueError('GPU admission incomplete for '+condition)
    if len(set(g.get('real_image_uids',[])))<20:
        raise ValueError('Need 20 unique real-image problems')
    if g.get('long_prefix_tokens',0)<4096 or g.get('near_cap_tokens',0)<32767:
        raise ValueError('Long/cap state evidence incomplete')
    if not g.get('evidence_files') or not g.get('numeric_tolerance_calibration'):
        raise ValueError('Evidence/tolerance baseline absent')
    return g

def validate_trace(trace, full_n, method):
    if len(trace.get('prefill_layers',{}))!=36 or len(trace.get('last_layers',{}))!=36:
        raise ValueError('Missing 36-layer state')
    selections=trace.get('selections',[])
    if method not in ('Full','Resolution') and not selections:raise ValueError('Selector never ran')
    for s in selections:
        ids=s['original_indices']
        if len(ids)!=len(set(ids)) or any(i<0 or i>=full_n for i in ids):
            raise ValueError('Invalid original token lineage')
    if selections and len(selections[-1]['original_indices'])!=trace['expected_final_visual_tokens']:
        raise ValueError('Measured final visual budget differs from protocol')
    return True
