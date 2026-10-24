from __future__ import annotations
import hashlib, json, os
from pathlib import Path

class Blocked(RuntimeError): pass

def digest(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,path)

def seed(uid,repeat):
    if repeat not in (0,1,2): raise ValueError('repeat must be 0,1,2')
    return (1234+int(digest(uid)[:8],16)+1000003*repeat)%(2**31-1)

def code_hash():
    root=Path(__file__).parent
    return digest(json.dumps({p.relative_to(root).as_posix():digest(p.read_bytes()) for p in sorted(root.rglob('*.py'))},sort_keys=True))

def require_gate(path,spec_sha,engine_sha=None):
    g=read(path)
    if not g.get('passed') or g.get('spec_sha256')!=spec_sha or g.get('code_sha256')!=code_hash():
        raise Blocked('Gate absent, failed or stale: '+str(path))
    if engine_sha is not None and g.get('engine_sha256')!=engine_sha: raise Blocked('Engine changed after certification')
    return g
