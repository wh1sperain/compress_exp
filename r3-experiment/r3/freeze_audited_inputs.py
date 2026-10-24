"""Join only complete, hash-verified native audits before deterministic sampling."""
import argparse,json
from pathlib import Path
from .common import read,write,digest,Blocked
from .manifest import freeze


def assemble(candidate_paths,audit_paths):
    candidates={};sources={};audit_hashes={};seen=set()
    for path in map(Path,candidate_paths):
        raw=path.read_bytes();sources[str(path)]=digest(raw)
        for line in raw.decode('utf-8').splitlines():
            row=json.loads(line);uid=row['uid']
            if uid in candidates:raise Blocked('Duplicate source UID')
            row['native_inputs']={};candidates[uid]=row
    for path in map(Path,audit_paths):
        status=read(path.with_suffix('.status.json'))
        if status.get('stage')!='complete' or status.get('exit_code')!=0:raise Blocked('Audit not complete: '+str(path))
        raw=path.read_bytes();sha=digest(raw)
        if sha!=status.get('audit_sha256'):raise Blocked('Audit hash changed')
        if status.get('candidate_sha256') and status['candidate_sha256'] not in sources.values():
            raise Blocked('Audit refers to different candidate source')
        rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
        if len(rows)!=status.get('total') or len(rows)!=status.get('completed'):raise Blocked('Incomplete audit rows')
        audit_hashes[str(path)]=sha
        for row in rows:
            uid,model=row['uid'],row['model'];key=(uid,model)
            if uid not in candidates or model not in {'llava7','qwen8','mini8'} or key in seen:
                raise Blocked('Unknown or duplicate audit identity')
            seen.add(key);candidates[uid]['native_inputs'][model]=row['audit']
    if any(set(r['native_inputs'])!={'llava7','qwen8','mini8'} for r in candidates.values()):
        raise Blocked('Missing model/candidate audit pairing')
    return list(candidates.values()),{'candidate_files':sources,'audit_files':audit_hashes,
                                    'candidate_count':len(candidates),'audit_count':len(seen)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidates',nargs='+',required=True)
    p.add_argument('--audits',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();rows,receipt=assemble(a.candidates,a.audits)
    freeze(rows,Path(a.out));write(Path(a.out)/'input_audit_lock.json',receipt)
