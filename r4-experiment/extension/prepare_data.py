"""Freeze normalized source records before generation; labels get a separate file."""
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from .common import *

FIELDS = ('sample_id', 'dataset', 'split', 'question', 'choices', 'gold_answer', 'question_type',
          'task_type', 'image_paths', 'image_count', 'image_hash', 'source_id')

def validate(row, verify_images=True):
    if any(k not in row for k in FIELDS): raise ValueError('Incomplete source record')
    if row['dataset'] not in DATASETS or row['split'] != SPLITS[row['dataset']]: raise ValueError('Wrong dataset/split')
    if not row['sample_id'].startswith(row['dataset']+':'+row['split']+':'): raise ValueError('Noncanonical sample_id')
    if not row['question_type'] or not row['task_type'] or not row['source_id']: raise ValueError('Missing strata/source')
    if not isinstance(row['choices'],list) or len(row['choices'])>26:raise ValueError('Choices must be an ordered list')
    if row['dataset']=='AI2D' and row['gold_answer'] not in [chr(65+i) for i in range(len(row['choices']))]:raise ValueError('Normalize AI2D source option index to an explicit letter before freeze')
    if row['dataset']=='MathVista' and (row['question_type'] not in ('multi_choice','free_form') or 'answer_type' not in row):raise ValueError('Preserve official MathVista question/answer type metadata')
    if row['image_count'] != 1 or len(row['image_paths']) != 1: return False
    if row['dataset'] == 'MathVerse' and row.get('variant') != 'Vision Dominant': return False
    if row['dataset'] == 'ChartQA' and row.get('subset') not in ('Human', 'Augmented'): raise ValueError('Missing ChartQA subset')
    if row['dataset'] == 'MMMU' and not row.get('discipline'): raise ValueError('Missing MMMU broad discipline')
    if len(row['image_hash']) != 64: raise ValueError('Missing original image SHA256')
    if verify_images and digest(Path(row['image_paths'][0]).read_bytes()) != row['image_hash']: raise ValueError('Image hash mismatch')
    return True

def stratum(row):
    d = row['dataset']
    return row['subset'] if d == 'ChartQA' else row['discipline'] if d == 'MMMU' else row['task_type'] if d == 'MathVista' else row['question_type']

def select(pool, count, quotas=None):
    """Round robin strata; prefer unseen images, then permit within-panel image reuse.

    ChartQA quotas are strict; MMMU quota shortages redistribute and are receipted.
    """
    bins = defaultdict(list)
    for r in sorted(pool, key=lambda x:x['sample_id']): bins[stratum(r)].append(r)
    rng = random.Random(42)
    for k in sorted(bins): rng.shuffle(bins[k])
    chosen=[]; seen_sources=set(); seen_images=set(); counts=Counter()
    while len(chosen) < count:
        progress=False
        for unique_image in (True, False):
            for k in sorted(bins):
                if quotas and counts[k] >= quotas.get(k,0): continue
                candidate=next((r for r in bins[k] if r['source_id'] not in seen_sources and
                               (not unique_image or r['image_hash'] not in seen_images)),None)
                if candidate is None: continue
                chosen.append(candidate); bins[k].remove(candidate); counts[k]+=1
                seen_sources.add(candidate['source_id']); seen_images.add(candidate['image_hash']); progress=True
                if len(chosen)==count: return chosen
            if progress: break
        if not progress: break
    return chosen

def freeze(candidates, out, source_lock, verify_images=True):
    if len({r['sample_id'] for r in candidates}) != len(candidates): raise ValueError('Duplicate sample_id')
    if not source_lock or any(d not in source_lock for d in DATASETS): raise ValueError('Six source versions required')
    for d in DATASETS:
        if not source_lock[d].get('revision') or not source_lock[d].get('source'): raise ValueError('Unpinned dataset source')
    eligible=[r for r in candidates if validate(r,verify_images)]
    pilot=[]; formal=[]; distributions={}
    # Select ALL pilot panels first, then exclude their image hashes across datasets.
    for d in DATASETS:
        p=[r for r in eligible if r['dataset']==d]
        q={'Human':10,'Augmented':10} if d=='ChartQA' else None
        selected=select(p,20,q)
        if len(selected)!=20: raise ValueError('Pilot shortage: '+d)
        pilot+=selected
    pilot_images={r['image_hash'] for r in pilot}
    for d in DATASETS:
        p=[r for r in eligible if r['dataset']==d and r['image_hash'] not in pilot_images and
           (d,r['source_id']) not in {(x['dataset'],x['source_id']) for x in pilot}]
        quotas={'Human':75,'Augmented':75} if d=='ChartQA' else None
        if d=='MMMU':
            disciplines=sorted({r['discipline'] for r in candidates if r['dataset']==d})
            if len(disciplines)!=6: raise ValueError('MMMU must use SIX broad disciplines, not 30 subjects')
            quotas={k:25 for k in disciplines}
        selected=select(p,150,quotas)
        if d=='MMMU' and len(selected)<150:
            used={r['source_id'] for r in selected}
            selected+=select([r for r in p if r['source_id'] not in used],150-len(selected))
        if len(selected)!=150: raise ValueError('Formal shortage: '+d)
        formal+=selected
        distributions[d]=dict(strata=dict(Counter(stratum(r) for r in selected)),
                              unique_images=len({r['image_hash'] for r in selected}))
    out=Path(out)
    if out.exists(): raise ValueError('Refuse to overwrite a frozen manifest')
    # Full manifest is the archival form. GPU only reads whitelisted inputs.
    save_rows(out/'manifest.jsonl',formal);save_rows(out/'pilot_manifest.jsonl',pilot)
    for name,data in [('formal',formal),('pilot',pilot)]:
        save_rows(out/(name+'_inputs.jsonl'),map(input_only,data))
        save_rows(out/(name+'_labels.jsonl'),({k:r[k] for k in ('sample_id','gold_answer','image_hash')} |
                  {k:r[k] for k in ('answer_type','precision','unit') if k in r} for r in data))
    receipt=dict(seed=42,formal=len(formal),pilot=len(pilot),source_lock=source_lock,
                 candidate_sha256=digest(candidates),distributions=distributions,
                 formal_inputs_sha256=digest((out/'formal_inputs.jsonl').read_bytes()),
                 pilot_inputs_sha256=digest((out/'pilot_inputs.jsonl').read_bytes()),
                 all_images_verified=verify_images,performance_filtering=False)
    save(out/'freeze_receipt.json',receipt);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidates',required=True)
    p.add_argument('--source-lock',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    freeze(rows(a.candidates),a.out,read(a.source_lock))
