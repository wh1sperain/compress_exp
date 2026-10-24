"""Monotone checkpoint publication; replay can never replace a longer prefix."""
from pathlib import Path
from .common import read, write, digest, Blocked


def publish(path, receipt, saved):
    path = Path(path)
    ids = receipt['generated_token_ids']
    if ids[:min(len(ids), len(saved))] != saved[:min(len(ids), len(saved))]:
        raise Blocked('Replayed prefix differs from immutable original')
    if len(ids) < len(saved):
        return False
    if path.exists():
        old = read(path)
        previous = old['generated_token_ids']
        if old['job_key'] != receipt['job_key'] or old['seed'] != receipt['seed']:
            raise Blocked('Checkpoint identity changed')
        if len(previous) > len(ids) or ids[:len(previous)] != previous:
            raise Blocked('Refuse checkpoint regression or prefix replacement')
        if len(previous) == len(ids):
            return False
        raw = path.read_bytes()
        history = path.with_name(path.name + '.history')
        history.mkdir(exist_ok=True)
        archive = history / (str(len(previous)) + '-' + digest(raw) + '.json')
        if not archive.exists():
            with archive.open('xb') as stream:
                stream.write(raw)
        elif archive.read_bytes() != raw:
            raise Blocked('Checkpoint history collision')
    write(path, receipt)
    return True
