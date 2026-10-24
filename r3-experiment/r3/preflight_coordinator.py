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
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--methods',nargs='+');a=p.parse_args();folder=Path(a.folder)
    lock=(folder/'preflight-coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=folder/'telemetry/preflight_state.json'
    state={'pid':os.getpid(),'production_started':False,'started_at':time.time(),'stage':'environment_install'}
    attempt=folder/'tests'/('native-smoke-'+str(time.time_ns()))
    attempt.mkdir(parents=True);state['attempt_directory']=str(attempt)
    write(status,state)
    try:
        registered=['Full','PriorTR50','PriorTR25','AgilePruner50','AgilePruner25','VisionZip50','VisionZip25','FastVFull','FastV50','FastV25','MiniNative','QwenFull','QwenPriorTR50','QwenPriorTR25']
        methods=a.methods or registered
        if not methods or any(m not in registered for m in methods):raise Blocked('Unregistered smoke method')
        tracks=sorted({'qwen' if m.startswith('Qwen') else 'mini' if m=='MiniNative' else 'fastv' if m.startswith('FastV') else 'llava' for m in methods})
        subprocess.run([sys.executable,'-m','r3.environments','--folder',str(folder),'--tracks',*tracks],check=True)
        environments=read(folder/'environment/native_environments.json')
        state['requested_methods']=methods
        for method in methods:
            required={'qwen8'} if method.startswith('Qwen') else {'mini8' if method=='MiniNative' else 'llava7','clip336'}
            while True:
                state.update(stage='waiting_for_verified_weights',method=method,required=sorted(required),updated_at=time.time());write(status,state)
                weights=folder/'manifests/model_weights_lock.json'
                if weights.exists() and required<=set(read(weights)):break
                bootstrap=read(folder/'telemetry/bootstrap_state.json')
                if bootstrap['stage']=='failed':raise Blocked('Required artifacts unavailable after preparation failure; original receipt retained')
                time.sleep(30)
            state.update(stage='native_synthetic_smoke',method=method,updated_at=time.time());write(status,state)
            env='qwen' if method.startswith('Qwen') else 'mini' if method=='MiniNative' else 'fastv' if method.startswith('FastV') else 'llava'
            module='r3.qwen_smoke' if env=='qwen' else 'r3.native_smoke'
            native_method=method.removeprefix('Qwen') if env=='qwen' else method
            subprocess.run([environments[env]['python'],'-m',module,'--folder',str(folder),'--method',native_method,'--out',str(attempt/('smoke-'+method+'.json'))],check=True)
        state.update(stage='smoke_complete_not_certified',exit_code=0,updated_at=time.time())
        write(status,state);write(folder/'reports/preflight_completion.json',state)
    except BaseException as exc:
        state.update(stage='failed',exit_code=1,updated_at=time.time(),error_type=type(exc).__name__,error=str(exc))
        write(status,state);write(folder/'failures'/('preflight-'+str(time.time_ns())+'.json'),state)
        raise


if __name__=='__main__':main()
