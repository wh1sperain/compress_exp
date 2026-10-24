"""Create separate native-library environments without replacing SM120 torch."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .common import read, write, Blocked


def prepare(folder, tracks):
    source=read(folder/'manifests/source_lock.json')
    receipt=folder/'environment/native_environments.json'
    result=read(receipt) if receipt.exists() else {}
    for track in tracks:
        env=folder/'environment'/('native-'+track)
        python=env/'bin/python'
        if not python.exists():subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(env)],check=True)
        requirements=['sentencepiece==0.2.1','shortuuid==1.0.13','einops==0.8.1','accelerate==0.29.0','huggingface-hub==0.36.0']
        if track=='llava':requirements+=['transformers==4.37.2','tokenizers==0.15.1']
        elif track=='mini':requirements+=['transformers==4.43.1','tokenizers==0.19.1']
        elif track=='fastv':requirements+=['transformers==4.39.0','tokenizers==0.15.2']
        elif track=='qwen':
            # Exact author-required revision; pip direct_url records full SHA.
            requirements=['git+https://github.com/huggingface/transformers.git@f8f2834e1a','accelerate==1.12.0']
        else:raise Blocked('Unknown isolated environment '+track)
        subprocess.run([str(python),'-m','pip','install','--index-url','https://pypi.org/simple',*requirements],check=True)
        packages=subprocess.check_output([str(python),'-m','pip','freeze'],text=True)
        (env/'frozen-packages.txt').write_text(packages)
        check='import torch,transformers; assert torch.cuda.is_available(); assert torch.version.cuda=="13.0"; print(torch.__version__,transformers.__version__)'
        actual=subprocess.check_output([str(python),'-c',check],text=True).strip()
        if track=='qwen':
            provenance=subprocess.check_output([str(python),'-c',
                'import importlib.metadata as m; print(m.distribution("transformers").read_text("direct_url.json"))'],text=True)
            origin=json.loads(provenance)
            if not origin.get('vcs_info',{}).get('commit_id','').startswith('f8f2834e1a'):
                raise Blocked('Qwen transformers source revision mismatch')
            write(env/'transformers_source.json',origin)
        result[track]={'python':str(python),'requirements':requirements,'actual':actual,'model_certified':False}
        write(folder/'environment/native_environments.partial.json',result)
    write(folder/'environment/native_environments.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',required=True);p.add_argument('--tracks',nargs='+',default=['llava','mini','fastv'])
    a=p.parse_args();prepare(Path(a.folder),a.tracks)


if __name__=='__main__':main()
