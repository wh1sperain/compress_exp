"""Import pinned official JSON exports through explicit field maps; never guess schemas.

Source map specifies six datasets' local official exports or HTTPS download URL,
SHA256, immutable revision, record path and exact mappings. Images stay in place.
"""
import argparse
import urllib.request
from .common import *

def at(obj,path):
    if path=='':return obj
    for k in path.split('.'):
        obj=obj[int(k)] if isinstance(obj,list) else obj[k]
    return obj

def import_sources(mapping,out):
    if set(mapping)!=set(DATASETS):raise ValueError('Need exactly six pinned source definitions')
    out=Path(out)
    if out.exists():raise ValueError('Source import output exists')
    normalized=[];locks={}
    for dataset,cfg in mapping.items():
        if not cfg.get('revision') or cfg['revision'] in ('main','HEAD','latest'):raise ValueError('Pin immutable source revision')
        p=Path(cfg['local_path'])
        if not p.exists():
            url=cfg.get('url','')
            if not url.startswith('https://'):raise ValueError('Missing local official export/HTTPS URL')
            req=urllib.request.Request(url,headers={'User-Agent':'visual-compression-research'})
            with urllib.request.urlopen(req,timeout=60) as response:raw=response.read()
            if digest(raw)!=cfg['sha256']:raise ValueError('Downloaded source SHA mismatch')
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(raw)
        raw=p.read_bytes()
        if digest(raw)!=cfg['sha256']:raise ValueError('Source SHA mismatch')
        data=json.loads(raw.decode('utf-8-sig'));data=at(data,cfg.get('records_path',''))
        if isinstance(data,dict):data=[dict(value,original_id=key) for key,value in data.items()]
        for original in data:
            r={k:at(original,path) for k,path in cfg['fields'].items()};r.update(cfg.get('constants',{}))
            r.update(dataset=dataset,split=SPLITS[dataset])
            r['sample_id']=dataset+':'+r['split']+':'+str(r.pop('original_id'))
            if isinstance(r['image_paths'],str):r['image_paths']=[r['image_paths']]
            r['image_paths']=[str((Path(cfg['image_root'])/x).resolve()) for x in r['image_paths']]
            r['image_count']=len(r['image_paths'])
            if r['image_count']!=1:continue
            r['image_hash']=digest(Path(r['image_paths'][0]).read_bytes())
            if 'value_maps' in cfg:
                for field,values in cfg['value_maps'].items():r[field]=values[str(r[field])]
            r['source_id']=str(r['source_id']);normalized.append(r)
        locks[dataset]={k:cfg[k] for k in ('revision','sha256')};locks[dataset]['source']=cfg.get('url',str(p))
    save_rows(out/'candidates.jsonl',normalized);save(out/'source_lock.json',locks)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-map',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();import_sources(read(a.source_map),a.out)
