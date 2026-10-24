"""Resolve immutable sources/configs; does NOT certify models or write a ready spec."""
from __future__ import annotations
import argparse, io, json, tarfile, urllib.request, posixpath
from pathlib import Path
from .common import read,write,digest,Blocked

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'compress-exp-r3-source-lock'})
    with urllib.request.urlopen(req,timeout=60) as r: return r.read()

def sources(spec,folder):
    lock={}
    frozen=read(Path(__file__).resolve().parent.parent/'source_revisions.json')
    previous=read(folder/'manifests/source_lock.json') if (folder/'manifests/source_lock.json').exists() else {}
    for name,item in spec['repositories'].items():
        repo=item['url'].removeprefix('https://github.com/').removesuffix('.git')
        sha=item['revision'] or previous.get(name,{}).get('revision') or frozen[name]
        meta=json.loads(get(f'https://api.github.com/repos/{repo}/commits/{sha or "HEAD"}'))
        sha=meta['sha']; assert len(sha)==40
        dest=folder/'vendor'/name/sha
        if not (dest/'r3_download_receipt.json').exists():
            blob=get(f'https://codeload.github.com/{repo}/tar.gz/{sha}')
            with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
                members=archive.getmembers()
                links=[]
                for m in members:
                    parts=Path(m.name).parts[1:]
                    if not parts: continue
                    if m.issym() or m.islnk():
                        links.append({'path':m.name,'target':m.linkname})
                        continue
                    target=dest.joinpath(*parts).resolve()
                    if not target.is_relative_to(dest.resolve()): raise Blocked('Unsafe archive path')
                    if m.isfile():
                        target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_bytes(archive.extractfile(m).read())
                # Materialize only file links resolving within this same archive.
                unresolved=[]
                for link in links:
                    resolved=posixpath.normpath(posixpath.join(posixpath.dirname(link['path']),link['target']))
                    try: member=archive.getmember(resolved)
                    except KeyError: unresolved.append(link);continue
                    if not member.isfile():unresolved.append(link);continue
                    target=dest.joinpath(*Path(link['path']).parts[1:]).resolve()
                    if not target.is_relative_to(dest.resolve()):raise Blocked('Unsafe link path')
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(member).read())
            write(dest/'r3_download_receipt.json',{'repository':repo,'commit':sha,'archive_sha256':digest(blob),'unresolved_symlinks':unresolved})
        files={p.relative_to(dest).as_posix():digest(p.read_bytes()) for p in sorted(dest.rglob('*')) if p.is_file()}
        lock[name]={'repository':repo,'revision':sha,'path':str(dest.resolve()),'files':files}
        write(folder/'manifests/source_lock.partial.json',lock)
        print(json.dumps({'source':name,'revision':sha,'files':len(files)}),flush=True)
    write(folder/'manifests/source_lock.json',lock)
    return lock

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--spec',required=True);p.add_argument('--root',required=True)
    p.add_argument('--sources-only',action='store_true');p.add_argument('--model-endpoint',default='https://huggingface.co');a=p.parse_args()
    spec=read(a.spec);folder=Path(a.root)/spec['run_subdir']
    for name in ['environment','manifests','configs','source_snapshot','tests','jobs','raw','metrics','annotations','telemetry','reports','failures']: (folder/name).mkdir(parents=True,exist_ok=True)
    lock=sources(spec,folder)
    unresolved=['native adapters and actual tensor audits','model shard hashes and CLIP revision','native input eligibility and 500/100 manifests','Mini loaded configuration and prefusion','native/engine certification']
    model_lock={}
    if not a.sources_only:
        previous_models=read(folder/'manifests/model_config_lock.json') if (folder/'manifests/model_config_lock.json').exists() else {}
        for key,m in spec['models'].items():
            revision=previous_models.get(key,{}).get('revision') or m.get('revision')
            meta=json.loads(get(a.model_endpoint+'/api/models/'+m['model_id']+(('/revision/'+revision) if revision else '')))
            sha=meta['sha'];blob=get(f'{a.model_endpoint}/{m["model_id"]}/resolve/{sha}/config.json')
            cfg=json.loads(blob);write(folder/'configs'/f'{key}.checkpoint.json',cfg)
            model_lock[key]={'model_id':m['model_id'],'revision':sha,'config_sha256':digest(blob),'transport_endpoint':a.model_endpoint,'weights_downloaded':False}
            write(folder/'manifests/model_config_lock.partial.json',model_lock)
        write(folder/'manifests/model_config_lock.json',model_lock)
    write(folder/'reports/preparation_state.json',{'production_ready':False,'sources_locked':list(lock),'model_configs_locked':list(model_lock),'unresolved':unresolved})
    # No resolved_spec is emitted before actual model/input/method validation.
    print(json.dumps({'production_ready':False,'unresolved':unresolved}))

if __name__=='__main__': main()
