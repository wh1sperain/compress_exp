import hashlib
import json
from pathlib import Path

TASKS = ('ChartQA-human-test', 'DocVQA', 'GQA-balanced-val', 'InfographicVQA', 'MathVista-testmini')
METHODS = ('FastV', 'PriorTR', 'VisPruner', 'SparseVLM', 'HiLo', 'Random', 'Resolution')
CONDITIONS = ('Full',) + tuple(f'{m}{r}' for m in METHODS for r in (50, 25))
ROOT = Path(__file__).resolve().parents[1]

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(x):
    if not isinstance(x, bytes):
        x = json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(x).hexdigest()

def write_once(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def lines(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def seed(uid, repeat):
    # Exactly R3's request RNG seed; selector RNG is a separate namespace.
    h = hashlib.sha256(uid.encode()).hexdigest()
    return (1234 + int(h[:8], 16) + 1000003 * repeat) % (2**31-1)

def code_hash():
    return sha({p.relative_to(ROOT).as_posix(): sha(p.read_bytes())
                for p in sorted((ROOT / 'r4').glob('*.py'))})

def condition(name):
    if name == 'Full':
        return 'Full', 1.
    if name not in CONDITIONS and name not in tuple(m+str(r) for m in METHODS[:5] for r in (90,75,10)):
        raise ValueError('Unknown condition: ' + name)
    return name[:-2], int(name[-2:]) / 100

def select(rows, per_task, salt, excluded_images=()):
    """Question sampling, never filtered on length, quality, or errors."""
    if len({r['uid'] for r in rows}) != len(rows):
        raise ValueError('Duplicate UID')
    out = []
    for task in TASKS:
        eligible = [r for r in rows if r['task'] == task and r['image_sha256'] not in excluded_images]
        eligible.sort(key=lambda r: sha([salt, task, r['uid']]))
        if len(eligible) < per_task:
            raise ValueError(f'Insufficient independent {task}: {len(eligible)}/{per_task}')
        out.extend(eligible[:per_task])
    return out

def input_only(row):
    required = ('uid', 'task', 'question', 'image_sha256', 'image_path')
    result = {k: row[k] for k in required}
    result['options'] = row.get('options') or []
    return result

def matrix(rows, spec, phase='R4-1', conditions=CONDITIONS):
    jobs = []
    for row in rows:
        for repeat in range(3):
            for name in conditions:
                job = dict(phase=phase, uid=row['uid'], task=row['task'], repeat=repeat,
                           seed=seed(row['uid'], repeat), condition=name,
                           input_sha256=sha(input_only(row)), protocol_sha256=sha(spec),
                           generation=spec['generation'])
                job['job_key'] = sha(job)
                jobs.append(job)
    return jobs
