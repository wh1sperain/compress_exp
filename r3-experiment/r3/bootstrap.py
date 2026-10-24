"""Isolated artifact preparation with exclusive lock and durable failure receipt.

This is preparation, never the formal experiment coordinator. No production
records or model certification are issued by this module.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from .common import read, write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True);p.add_argument('--root',required=True)
    a=p.parse_args();spec=read(a.spec);folder=Path(a.root)/spec['run_subdir']
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'bootstrap.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another R3 preparation process is active')
    state={'pid':os.getpid(),'started_at':time.time(),'production_started':False,'stage':'initializing'}
    status=folder/'telemetry/bootstrap_state.json'
    def step(name,cmd):
        state.update(stage=name,updated_at=time.time());write(status,state)
        subprocess.run(cmd,check=True)
    try:
        for d in ('environment','manifests','configs','tests','telemetry','failures','reports'):(folder/d).mkdir(exist_ok=True)
        step('source_lock',[sys.executable,'-m','r3.prepare','--spec',a.spec,'--root',a.root])
        step('selector_component_probe',[sys.executable,'-m','r3.selector_probe','--source-lock',str(folder/'manifests/source_lock.json'),'--out',str(folder/'tests/selector_components.json')])
        # Separate environment shares only the existing SM120-capable torch.
        # No torch downgrade, project setup.py, or global package edits.
        env=folder/'environment/artifacts-venv'
        if not (env/'bin/python').exists():step('create_artifact_environment',[sys.executable,'-m','venv','--system-site-packages',str(env)])
        python=str(env/'bin/python')
        step('install_artifact_dependencies',[python,'-m','pip','install','huggingface-hub==0.36.0'])
        with (folder/'environment/artifact_pip_freeze.txt').open('w') as out:
            subprocess.run([python,'-m','pip','freeze'],stdout=out,check=True)
        step('download_model_weights',[python,'-m','r3.artifacts','--spec',a.spec,'--root',a.root])
        state.update(stage='artifacts_ready',updated_at=time.time(),exit_code=0,
                     remaining=['native model adapters and input manifests','Full/native/engine gates','formal experiment launch'])
        write(status,state);write(folder/'reports/bootstrap_completion.json',state)
    except BaseException as exc:
        state.update(stage='failed',updated_at=time.time(),exit_code=1,error_type=type(exc).__name__,error=str(exc))
        write(status,state);write(folder/'failures'/('bootstrap-'+str(time.time_ns())+'.json'),state)
        raise


if __name__=='__main__':main()
