"""Download code only at locked commits; never credentials, datasets or checkpoints."""
import argparse
import subprocess
from pathlib import Path
from .core import read
from .adapters import verify_source

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--locks',default='sources.lock.json');p.add_argument('--out',default='vendor')
    a=p.parse_args()
    for name,entry in read(a.locks).items():
        if 'url' not in entry:continue
        target=Path(a.out)/name
        if not target.exists():
            target.mkdir(parents=True)
            subprocess.run(['git','init',str(target)],check=True)
            subprocess.run(['git','-C',str(target),'remote','add','origin',entry['url']],check=True)
            subprocess.run(['git','-C',str(target),'fetch','--depth','1','origin',entry['revision']],check=True)
            subprocess.run(['git','-C',str(target),'checkout','--detach','FETCH_HEAD'],check=True)
        verify_source(target,entry['revision'])
