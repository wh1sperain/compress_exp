"""Label-independent, exact-budget intervention construction; no model answer correction."""
import math
import random
from .core import sha

def choose(population, k, key):
    return random.Random(int(sha(key),16)).sample(sorted(population),k)

def rescue_pair(selected, evidence, total, key):
    S,G=set(selected),set(evidence)
    if len(S)!=len(selected) or not S or not G or not S|G <= set(range(total)):
        raise ValueError('Invalid selected/evidence indices')
    add=G-S
    removable=S-G
    placebo_pool=set(range(total))-S-G
    k=len(add)
    if not k or len(removable)<k or len(placebo_pool)<k:
        raise ValueError('Not eligible for same-budget matched rescue; do not truncate G')
    removed=set(choose(removable,k,[key,'remove']))
    placebo=set(choose(placebo_pool,k,[key,'placebo']))
    return dict(original=sorted(S),removed=sorted(removed),added_evidence=sorted(add),
        added_placebo=sorted(placebo),rescue=sorted((S-removed)|add),
        placebo=sorted((S-removed)|placebo),budget=len(S))

def map_region(region, token_boxes, image_sha, mapping_receipt):
    """Consume audited actual preprocessing geometry; NEVER invent a uniform original-image grid."""
    if mapping_receipt.get('image_sha256')!=image_sha or not mapping_receipt.get('processor_mapping_verified'):
        raise ValueError('Actual preprocessing mapping absent')
    if mapping_receipt.get('token_boxes_sha256')!=sha(token_boxes):
        raise ValueError('Mapping changed')
    x0,y0,x1,y1=region
    if not (0<=x0<x1<=1 and 0<=y0<y1<=1):raise ValueError('Normalized ROI invalid')
    return [i for i,(a,b,c,d) in enumerate(token_boxes) if max(a,x0)<min(c,x1) and max(b,y0)<min(d,y1)]

def prefix_triplet(tokens, start, end, correction, neutral, max_budget=32768):
    if not 0<=start<end<=len(tokens) or not correction or not neutral:
        raise ValueError('Invalid first-error token span')
    # Cut at the first error, never replay later erroneous reasoning.
    original=list(tokens[:end]);fixed=list(tokens[:start])+list(correction)
    control=list(tokens[:start])+list(neutral)
    if abs(len(correction)-len(neutral))>2:raise ValueError('Neutral edit not length matched')
    if max(map(len,(original,fixed,control)))>=max_budget:raise ValueError('No continuation budget')
    return {name:dict(forced_prefix=p, rng_offset=end, max_new_tokens=max_budget-len(p))
            for name,p in [('original',original),('corrected',fixed),('neutral',control)]}

def stop_prefix(tokens, budget, think_close=151668, eos_ids=(151645,), structural_ids=(151667,)):
    """Budget counts reasoning tokens; natural earlier closure reuses entire original output."""
    if budget not in (256,512,1024):raise ValueError('Unregistered stop budget')
    p=[];count=0
    for token in tokens:
        if token==think_close or token in eos_ids:return dict(reuse_natural=True)
        if count==budget:break
        p.append(token)
        if token not in structural_ids:count+=1
    if count<budget:raise ValueError('Incomplete natural prefix')
    return dict(reuse_natural=False,forced_prefix=p+[think_close],rng_offset=len(p),
                max_new_tokens=32768-len(p)-1,answer_budget_in_total_cap=True)

def verified_fact_pair(fact,neutral,tokenizer,approval):
    if len(set(approval.get('reviewers',[])))<2 or not approval.get('no_final_answer_leakage'):
        raise ValueError('Two independent human approvals required for fact/no-answer check')
    if len(tokenizer.encode(fact,add_special_tokens=False))!=len(tokenizer.encode(neutral,add_special_tokens=False)):
        raise ValueError('Fact/neutral must match actual tokenizer length')
    return {'fact':fact,'neutral':neutral,'approval':approval}
