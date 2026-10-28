"""Evidence validation only: never manufactures GPU PASS from CPU tests."""
import argparse
from .common import *

CHECKS=('loading_bf16','no_pruning_logits','no_pruning_output','actual_retention',
        'text_preserved','positions','preprocessing','tensor_shapes','think_parser',
        'evaluation_pipeline','determinism','full_output_budget')

def admit(receipt, identity, model, condition):
    if receipt.get('identity')!=identity: raise ValueError('Unbound runtime evidence')
    g=receipt.get('conditions',{}).get(model+':'+condition,{})
    if any(g.get('checks',{}).get(k)!='PASS' for k in CHECKS): raise ValueError('GPU/evaluation gate incomplete')
    if not g.get('evidence_files') or not g.get('tolerances'): raise ValueError('No measured evidence/tolerances')
    for f in g['evidence_files']:
        if digest(Path(f['path']).read_bytes())!=f['sha256']: raise ValueError('Changed gate evidence')
    return g

def greedy_decision(records, manifest, confirmations):
    ids={r['sample_id'] for r in manifest}
    if len(ids)!=120 or any(sum(r['dataset']==d for r in manifest)!=20 for d in DATASETS): raise ValueError('Expected 6 x 20 pilot inputs')
    result={}
    for model in MODELS:
        rr=[r for r in records if r['model']==model]
        if len(rr)!=120 or {r['sample_id'] for r in rr}!=ids: raise ValueError('Incomplete precheck model panel')
        if any(r['method']!='Full' or r['decoding_mode']!='greedy' or r['phase']!='precheck' for r in rr): raise ValueError('Wrong precheck phase/config')
        if any(r.get('run_status')!='success' or r.get('correct') is None for r in rr): raise ValueError('Resolve process failures/evaluation before decision')
        flagged=[r for r in rr if r['repetition_warning']['flag']]
        for r in flagged:
            c=confirmations.get(r['run_id'],{})
            if c.get('raw_sha256')!=r['raw_sha256'] or not c.get('reviewer') or type(c.get('confirmed')) is not bool: raise ValueError('Loop warning needs bound human confirmation')
        cap=sum(r['total_output_tokens']>=32768 for r in rr)/120
        loops=sum(confirmations[r['run_id']]['confirmed'] for r in flagged)
        result[model]=dict(mode='sampling' if cap>.05 or loops else 'greedy',cap_rate=cap,confirmed_loops=loops,
                           evidence_sha256=digest(rr),manifest_sha256=digest(manifest),formal_results_seen=False,
                           multiseed_replanning_required=bool(cap>.05 or loops))
    return result

def determinism(records):
    groups={}
    for r in records:
        k=(r['model'],r['sample_id'],r['method'],r['keep_ratio_target']);groups.setdefault(k,[]).append(r)
    out=[]
    for k,rr in groups.items():
        if len(rr)!=3 or {r['repeat'] for r in rr}!={0,1,2}: raise ValueError('Need three separately executed fixed-seed repeats')
        fields=('selected_visual_token_indices','visual_tokens_retained','raw_generated_token_ids','final_answer_text','reasoning_tokens')
        diff=[f for f in fields if len({digest(r[f]) for r in rr})>1]
        out.append(dict(configuration=k,status='PASS' if not diff else 'FAIL',different_fields=diff,
                        attribution='requires selector/operator/decoder diagnosis' if diff else None))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--manifest');p.add_argument('--confirmations')
    p.add_argument('--determinism',action='store_true');p.add_argument('--out',required=True);a=p.parse_args()
    value=determinism(rows(a.records)) if a.determinism else greedy_decision(rows(a.records),rows(a.manifest),read(a.confirmations))
    save(a.out,value)
