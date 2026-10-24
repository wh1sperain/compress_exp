"""Prepare isolated native environments and wait for verified weights for smoke.

Runs no research jobs. Full certification and data preparation remain separate.
"""
from __future__ import annotations
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from .common import read,write,Blocked


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder)
    lock=(folder/'preflight-coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=folder/'telemetry/preflight_state.json'
    state={'pid':os.getpid(),'production_started':False,'started_at':time.time(),'stage':'environment_install'}
    write(status,state)
    try:
        subprocess.run([sys.executable,'-m','r3.environments','--folder',str(folder)],check=True)
        environments=read(folder/'environment/native_environments.json')
        methods=['Full','PriorTR50','PriorTR25','AgilePruner50','AgilePruner25','VisionZip50','VisionZip25','FastV50','FastV25','MiniNative']
        for method in methods:
            required={'mini8' if method=='MiniNative' else 'llava7','clip336'}
            while True:
                state.update(stage='waiting_for_verified_weights',method=method,required=sorted(required),updated_at=time.time());write(status,state)
                weights=folder/'manifests/model_weights_lock.json'
                if weights.exists() and required<=set(read(weights)):break
                bootstrap=read(folder/'telemetry/bootstrap_state.json')
                if bootstrap['stage']=='failed':raise Blocked('Required artifacts unavailable after preparation failure; original receipt retained')
                time.sleep(30)
            state.update(stage='native_synthetic_smoke',method=method,updated_at=time.time());write(status,state)
            env='mini' if method=='MiniNative' else 'fastv' if method.startswith('FastV') else 'llava'
            subprocess.run([environments[env]['python'],'-m','r3.native_smoke','--folder',str(folder),'--method',method,'--out',str(folder/'tests'/('smoke-'+method+'.json'))],check=True)
        state.update(stage='smoke_complete_not_certified',exit_code=0,updated_at=time.time())
        write(status,state);write(folder/'reports/preflight_completion.json',state)
    except BaseException as exc:
        state.update(stage='failed',exit_code=1,updated_at=time.time(),error_type=type(exc).__name__,error=str(exc))
        write(status,state);write(folder/'failures'/('preflight-'+str(time.time_ns())+'.json'),state)
        raise


if __name__=='__main__':main()
