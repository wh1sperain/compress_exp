import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('ChartQA', 'MMMU', 'MathVista', 'MathVerse', 'GQA', 'AI2D')
SPLITS = dict(zip(DATASETS, ('test', 'validation', 'testmini', 'testmini', 'val_balanced', 'test')))
MODELS = ('qwen4', 'qwen8')
METHODS = ('PriorTR', 'FastV', 'VisPruner', 'SparseVLM', 'HiLo')
RATIOS = (75, 50, 25, 90, 10)

def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def save_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        for value in values: f.write(json.dumps(value, ensure_ascii=False) + '\n')

def code_hash():
    return digest({p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for folder in ('extension', 'r4') for p in sorted((ROOT/folder).glob('*.py'))})

def input_only(row):
    keys = ('sample_id', 'dataset', 'split', 'question', 'choices', 'question_type', 'task_type',
            'image_paths', 'image_count', 'image_hash', 'source_id', 'subset', 'discipline', 'variant')
    return {k: row[k] for k in keys if k in row}

def generation(mode):
    if mode not in ('greedy', 'sampling'): raise ValueError('Unknown decoding mode')
    g = dict(do_sample=mode == 'sampling', num_beams=1, num_return_sequences=1,
             max_new_tokens=32768, repetition_penalty=1.0, presence_penalty=0,
             batch_size=1, thinking=True, temperature=0.0 if mode == 'greedy' else 1.0)
    if mode == 'sampling': g.update(top_p=.95, top_k=20)
    return g

def model_spec(spec, model):
    return dict(spec, model_id=spec['models'][model]['model_id'], model_revision=spec['models'][model]['revision'])
