"""Deterministic image-level sampling on prevalidated inputs, no outputs read."""
from .common import digest,Blocked

TASKS=['DocVQA','InfographicVQA','ChartQA-human-test','GQA-balanced-val','MathVista-testmini']
def select(candidates):
    seen=set();main=[];cal=[];excluded=[]
    for task in TASKS:
        grouped={}
        for r in candidates:
            if r['task']!=task:continue
            if not r.get('question','').strip() or not r.get('rgb_decodable') or not r.get('single_image') or not r.get('public_label'):
                excluded.append({'uid':r['uid'],'reason':'candidate_eligibility'});continue
            if set(r.get('native_inputs',{}))!={'llava7','qwen8','mini8'} or not all(x.get('eligible') for x in r['native_inputs'].values()):
                excluded.append({'uid':r['uid'],'reason':'native_input_eligibility'});continue
            h=r['image_sha256']
            if h in seen:excluded.append({'uid':r['uid'],'reason':'cross_task_duplicate'});continue
            grouped.setdefault(h,[]).append(r)
        # Cross-task ownership follows task order, including images not sampled.
        seen.update(grouped)
        chosen=[min(rs,key=lambda r:digest('r2-question|'+r['uid'])) for rs in grouped.values()]
        chosen.sort(key=lambda r:digest('r2-image|'+r['image_sha256']))
        if len(chosen)<120:raise Blocked(f'{task}: only {len(chosen)} eligible unique images')
        main.extend(chosen[:100]);cal.extend(chosen[100:120])
    return main,cal,excluded

def split_labels(rows):
    labels={};workers=[]
    for r in rows:
        labels[r['uid']]={k:r[k] for k in ['answer','answers','label'] if k in r}
        if not labels[r['uid']]:raise Blocked('Missing label: '+r['uid'])
        workers.append({k:v for k,v in r.items() if k not in {'answer','answers','label','public_label'}})
    return workers,labels
