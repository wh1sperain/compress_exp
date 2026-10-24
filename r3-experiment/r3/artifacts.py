"""Download frozen model artifacts into R3 only; no old experiment mutation."""
from __future__ import annotations
import argparse
import hashlib
import os
import shutil
from pathlib import Path
from .common import Blocked, read, write


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def download_models(spec, folder):
    # AutoDL's proxy supports regular HTTPS but the Xet CAS download failed.
    # Keep the failed Xet partial files untouched; use a separate HTTP cache.
    os.environ['HF_HUB_DISABLE_XET']='1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(endpoint='https://huggingface.co')
    target = folder/'models'
    lock_path = folder/'manifests/model_weights_lock.json'
    locked = read(lock_path) if lock_path.exists() else {}
    config_lock = read(folder/'manifests/model_config_lock.json')
    items = {k: dict(v) for k, v in spec['models'].items()}
    items['clip336'] = {'model_id': 'openai/clip-vit-large-patch14-336', 'revision': None}
    for key, model in items.items():
        revision = locked.get(key, {}).get('revision') or config_lock.get(key, {}).get('revision') or model.get('revision')
        info = api.model_info(model['model_id'], revision=revision, files_metadata=True)
        revision = info.sha
        names = [f.rfilename for f in info.siblings]
        safetensors = any(n.endswith('.safetensors') for n in names)
        weight_ext = '.safetensors' if safetensors else '.bin'
        wanted = [n for n in names if '/' not in n and
                  (n.endswith(weight_ext) or n.endswith(('.json', '.model', '.txt', '.tiktoken')))]
        if not any(n.endswith(weight_ext) for n in wanted):
            raise Blocked('No frozen model weights: '+key)
        dest = target/key/revision/'https'
        sizes = {f.rfilename: f.size for f in info.siblings if f.rfilename in wanted}
        needed = sum(size or 0 for name, size in sizes.items() if not (dest/name).exists())
        if shutil.disk_usage(folder).free < needed+10*1024**3:
            raise Blocked('Insufficient space; old files retained: '+key)
        write(folder/'telemetry/artifact_status.json', {'stage': 'downloading', 'model': key, 'revision': revision, 'production_started': False})
        snapshot_download(model['model_id'], revision=revision, local_dir=dest,
                          allow_patterns=wanted, max_workers=4, endpoint='https://huggingface.co')
        files = {}
        for name in wanted:
            path = dest/name
            actual = sha256(path)
            item = next(f for f in info.siblings if f.rfilename == name)
            lfs = getattr(item, 'lfs', None)
            expected = getattr(lfs, 'sha256', None) if lfs is not None else None
            if isinstance(lfs, dict): expected = lfs.get('sha256')
            if (sizes[name] is not None and path.stat().st_size != sizes[name]) or (expected and actual != expected):
                raise Blocked('Artifact integrity failure: '+key+'/'+name)
            files[name] = {'sha256': actual, 'bytes': path.stat().st_size, 'lfs_sha256_verified': bool(expected)}
        locked[key] = {'model_id': model['model_id'], 'revision': revision, 'path': str(dest), 'files': files}
        write(lock_path, locked)
        print('model_artifacts_verified '+key, flush=True)
    write(folder/'telemetry/artifact_status.json', {'stage': 'complete', 'models': list(locked), 'production_started': False})
    return locked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True); p.add_argument('--root', required=True)
    a = p.parse_args(); spec = read(a.spec)
    download_models(spec, Path(a.root)/spec['run_subdir'])


if __name__ == '__main__': main()
