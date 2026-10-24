"""Deterministic image-level sampling on prevalidated inputs, no outputs read."""
from .common import digest,Blocked

TASKS=['DocVQA','InfographicVQA','ChartQA-human-test','GQA-balanced-val','MathVista-testmini']
def select(candidates):
    if len({r['uid'] for r in candidates})!=len(candidates):raise Blocked('Duplicate candidate uid')
    if any(r['task'] not in TASKS for r in candidates):raise Blocked('Unknown candidate task')
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


def freeze(candidates,out):
    """Publish once, after native input audits; never infer eligibility here."""
    import json
    from pathlib import Path
    from .common import write
    out=Path(out)
    if out.exists():raise Blocked('Frozen/partial manifest directory already exists')
    # Whitelist worker fields, rather than trusting arbitrary source metadata
    # not to contain hidden answers/rationales under another name.
    fields={'uid','task','image_sha256','image_path','question','options','answer_type',
            'precision','source_revision','source_file','native_inputs'}
    for r in candidates:
        if not r.get('source_revision') or not r.get('image_path'):
            raise Blocked('Missing source/image provenance: '+r['uid'])
        for model,audit in r.get('native_inputs',{}).items():
            if not audit.get('eligible'):continue
            required={'processor_sha256','prompt_sha256','expanded_prompt_tokens','visual_tokens','processed_size'}
            if not required<=audit.keys():raise Blocked('Incomplete native audit: '+r['uid']+'/'+model)
            if audit['expanded_prompt_tokens']<=0 or audit['visual_tokens']<=0:
                raise Blocked('Invalid native token count')
            limit=2048 if model=='llava7' else 8192 if model=='qwen8' else audit.get('native_context',0)-2048
            if audit['expanded_prompt_tokens']>limit:raise Blocked('Eligible audit exceeds context')
            if model=='llava7' and audit['visual_tokens']!=576:raise Blocked('Wrong A full visual count')
    main,cal,excluded=select(candidates)
    all_rows=main+cal
    _,labels=split_labels(all_rows)
    out.mkdir(parents=True)
    hashes={}
    for name,rows in [('main_manifest.jsonl',main),('calibration_manifest.jsonl',cal)]:
        safe=[]
        audit_fields={'eligible','processor_sha256','tokenizer_sha256','chat_template_sha256','prompt_sha256','expanded_prompt_tokens','visual_tokens','processed_size','native_context'}
        for r in rows:
            worker={k:r[k] for k in sorted(fields) if k in r}
            worker['native_inputs']={m:{k:v for k,v in a.items() if k in audit_fields} for m,a in r['native_inputs'].items()}
            safe.append(worker)
        payload=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in safe)
        (out/name).write_text(payload,encoding='utf-8');hashes[name]=digest(payload)
    write(out/'scoring_labels.json',labels)
    hashes['scoring_labels.json']=digest((out/'scoring_labels.json').read_bytes())
    selected={r['uid'] for r in all_rows}
    write(out/'manifest_diff.json',{'excluded':excluded,'not_selected':[r['uid'] for r in candidates if r['uid'] not in selected]})
    write(out/'manifest_lock.json',{'main':len(main),'calibration':len(cal),'hashes':hashes,
        'candidate_sha256':digest(json.dumps(candidates,sort_keys=True,ensure_ascii=False)),
        'production_ready':False,'scope':'input selection only; model/backend gates still required'})


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--audited-candidates',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();freeze([json.loads(x) for x in Path(a.audited_candidates).read_text(encoding='utf-8').splitlines()],a.out)
