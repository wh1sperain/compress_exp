#!/usr/bin/env python3
"""RTX PRO6000/local model-config preflight; no downloads, installs, or model weights loaded.

Example: python preflight.py --config llava7=/models/llava/config.json --out preflight.json
Hardware absence is reported as blocked, never as a successful GPU test.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any
from validate_plan import kv_bytes_per_token


def command(args: list[str]) -> dict[str, Any]:
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=15,check=False)
        return {'argv':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {'argv':args,'error':str(e)}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',action='append',default=[],metavar='NAME=/local/config.json')
    parser.add_argument('--out',type=Path,default=Path('preflight.json'))
    parser.add_argument('--gpu-index',type=int,default=0)
    args=parser.parse_args()
    blocked: list[str]=[]
    report: dict[str,Any]={'test_scope':'hardware, local config arithmetic, small BF16 GEMM only',
        'model_inference_run':False,'native_method_gate_passed':False,'kernel_and_natural_gate_passed':False,
        'python':platform.python_version(),'platform':platform.platform(),'cpu_logical_count':os.cpu_count(),
        'disk_free_bytes':shutil.disk_usage(Path.cwd()).free,'configs':{},'packages':{}}
    for package in ('torch','transformers','flashinfer-python','flash-attn','triton','numpy'):
        try: report['packages'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: report['packages'][package]=None
    if Path('/proc/meminfo').exists():
        report['meminfo']=Path('/proc/meminfo').read_text()
    report['nvidia_smi']=command(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.free,driver_version,power.limit,temperature.gpu','--format=csv,noheader,nounits'])
    report['gpu_processes']=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'])
    report['mig']=command(['nvidia-smi','--query-gpu=mig.mode.current','--format=csv,noheader'])
    for item in args.config:
        if '=' not in item:
            blocked.append('Invalid --config entry; expected NAME=PATH: '+item); continue
        name,path_text=item.split('=',1)
        try:
            path=Path(path_text).expanduser().resolve(strict=True)
            raw=path.read_bytes(); cfg=json.loads(raw)
            text_cfg=cfg.get('text_config',cfg)
            bpt=kv_bytes_per_token(cfg)
            report['configs'][name]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
                'model_type':cfg.get('model_type'),'kv_bytes_per_token_bf16':bpt,
                'kv_gib_at_4096':bpt*4096/2**30,'kv_gib_at_32768':bpt*32768/2**30,
                'context_config_value':text_cfg.get('max_position_embeddings'),
                'stored_dtype':text_cfg.get('dtype',cfg.get('torch_dtype')),
                'note':'KV estimate excludes weights, temporary activations, padding, page fragmentation and prefusion caches'}
        except (OSError,ValueError,KeyError,TypeError,ZeroDivisionError) as e:
            blocked.append(f'Config {name}: {e}')
    try:
        import torch
        report['torch_cuda_version']=torch.version.cuda
        report['cuda_available']=torch.cuda.is_available()
        if not torch.cuda.is_available():
            blocked.append('No CUDA GPU visible')
        else:
            i=args.gpu_index
            if i<0 or i>=torch.cuda.device_count(): raise ValueError('Invalid gpu-index')
            torch.cuda.set_device(i)
            props=torch.cuda.get_device_properties(i)
            cap=torch.cuda.get_device_capability(i)
            report['gpu']={'index':i,'name':props.name,'total_memory_bytes':props.total_memory,
                'total_memory_gib':props.total_memory/2**30,'compute_capability':list(cap),
                'bf16_supported':torch.cuda.is_bf16_supported(),'torch_arch_list':torch.cuda.get_arch_list()}
            if 'RTX PRO 6000' not in props.name.upper(): blocked.append('Not the planned RTX PRO 6000 device')
            if tuple(cap)!=(12,0): blocked.append('Compute capability differs from planned SM120')
            if props.total_memory/2**30<88: blocked.append('Visible memory below full-card 96GB-class plan; check SKU/MIG')
            if not torch.cuda.is_bf16_supported(): blocked.append('BF16 not supported by active stack')
            if not blocked:
                with torch.inference_mode():
                    x=torch.randn(256,256,device=f'cuda:{i}',dtype=torch.bfloat16)
                    y=x@x.T
                    torch.cuda.synchronize(i)
                    ok=bool(torch.isfinite(y).all().item())
                    report['small_bf16_gemm_finite']=ok
                    if not ok: blocked.append('Small BF16 GEMM produced nonfinite results')
                del x,y
    except (ImportError,RuntimeError,ValueError) as e:
        blocked.append('CUDA/PyTorch check failed: '+str(e))
    report['blocking_findings']=blocked
    report['preflight_status']='blocked' if blocked else 'hardware_preflight_passed_only'
    report['production_certified']=False
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['preflight_status'],'out':str(args.out),'findings':blocked},ensure_ascii=False))
    return 2 if blocked else 0
if __name__=='__main__':
    raise SystemExit(main())
