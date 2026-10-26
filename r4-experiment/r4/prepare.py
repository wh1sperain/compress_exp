"""Freeze label-free panels and raw annotation worklists. No inference/API calls."""
import argparse
from pathlib import Path
from .core import *

def prepare(source, out, spec, excluded=None, confirm=False, selected_methods=None):
    rows = lines(source)
    if confirm and excluded is None:
        raise ValueError('Confirmation requires ALL R3 image hashes, not just R4-1')
    excluded_hashes = {r['image_sha256'] for r in lines(excluded)} if excluded else set()
    if confirm:
        excluded_hashes.update(read(ROOT/'r3_exclusion_images.json'))
    n = 50 if confirm else 30
    panel = select(rows, n, spec['confirmation_salt'] if confirm else spec['discovery_salt'], excluded_hashes)
    panel = [input_only(r) for r in panel]
    if confirm:
        if not selected_methods or len(set(selected_methods)) != 3 or any(x not in METHODS[:5] for x in selected_methods):
            raise ValueError('Freeze three algorithm names before opening confirmation results')
        conditions = ('Full',) + tuple(f'{m}{r}' for m in selected_methods for r in (50,25))
    else:
        conditions = CONDITIONS
    jobs = matrix(panel, spec, 'R4-2' if confirm else 'R4-1', conditions)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_once(out/'panel.json', panel)
    write_once(out/'jobs.json', jobs)
    worklist = [dict(uid=r['uid'], image_sha256=r['image_sha256'], image_path=r['image_path'],
                     question=r['question'], options=r['options'],
                     facts=[], evidence_regions=[], reviewers=[], adjudicated=False)
                for r in panel]
    write_once(out/'image_fact_worklist.json', worklist)
    write_once(out/'receipt.json', dict(source_sha256=sha(Path(source).read_bytes()),
        excluded_source_sha256=sha(Path(excluded).read_bytes()) if excluded else None,
        panel_sha256=sha(panel), protocol_sha256=sha(spec), questions=len(panel), jobs=len(jobs),
        labels_in_generation=False, human_annotation_complete=False, formal_ready=False))
    return len(jobs)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True);p.add_argument('--out',required=True)
    p.add_argument('--spec',default='protocol.json');p.add_argument('--exclude-r3')
    p.add_argument('--confirm',action='store_true');p.add_argument('--methods',nargs=3)
    a=p.parse_args();print(prepare(a.source,a.out,read(a.spec),a.exclude_r3,a.confirm,a.methods))
