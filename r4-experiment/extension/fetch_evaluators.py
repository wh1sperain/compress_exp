"""Fetch only pinned author evaluator source files, never invoke their paid APIs."""
import argparse
import urllib.request
from .common import *

def fetch(lock,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);runtime={}
    for name,cfg in lock.items():
        url='https://raw.githubusercontent.com/'+cfg['repository']+'/'+cfg['revision']+'/'+cfg['path']
        p=out/(name+'.py')
        if p.exists():raw=p.read_bytes()
        else:
            with urllib.request.urlopen(url,timeout=30) as response:raw=response.read()
            if digest(raw)!=cfg['sha256']:raise ValueError('Author file SHA differs: '+name)
            with p.open('xb') as f:f.write(raw)
        if digest(raw)!=cfg['sha256']:raise ValueError('Cached author file changed: '+name)
        runtime[name]=dict(cfg,local_path=str(p.resolve()))
    save(out/'runtime_evaluators.json',runtime)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',default='evaluation_sources.lock.json');p.add_argument('--out',required=True)
    a=p.parse_args();fetch(read(a.lock),a.out)
