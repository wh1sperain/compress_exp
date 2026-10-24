"""Archive formal raw outputs and provenance; never score or rewrite generations.

Only explicitly selected result/registration trees and allowlisted small
provenance are read. Model, original image, dataset, vendor and environment
contents are never followed. Complete requires the registered 21900 jobs;
snapshots remain explicitly incomplete even if their count happens to match.
"""
import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path
from .common import Blocked,read,write,digest
from .runner import canonical

EXPECTED_RECORDS=21900
FORBIDDEN={'models','weights','vendor','environment','data','prepared_features','.git'}
PROVENANCE_NAMES={
    'manifests/source_lock.json','manifests/model_weights_lock.json',
    'manifests/frozen-inputs-v1/main_manifest.jsonl','manifests/frozen-inputs-v1/calibration_manifest.jsonl',
    'manifests/frozen-inputs-v1/scoring_labels.json','manifests/frozen-inputs-v1/manifest_lock.json',
    'manifests/frozen-inputs-v1/input_audit_lock.json','manifests/frozen-inputs-v1/manifest_diff.json',
}
PLAN_REFERENCES={'spec_path','manifest_path','engine_profile_path','native_gate_path','engine_gate_path',
                 'supplemental_audit_path','supplemental_status_path','execution_acceptance_path'}


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def checked_under(path,root):
    path=Path(path);root=Path(root).resolve()
    if path.is_symlink():raise Blocked('Symlink artifacts are forbidden')
    resolved=path.resolve()
    if not resolved.is_relative_to(root):raise Blocked('Artifact escapes selected root')
    # Reject a symlink parent even when its final target stays inside the root.
    for part in (path,*path.parents):
        if part==root:break
        if part.is_symlink():raise Blocked('Symlink ancestor forbidden')
    return resolved


def artifact_tree(root):
    root=Path(root).resolve();files=[]
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if path.is_symlink():raise Blocked('Symlink in artifact tree')
        if any(p in FORBIDDEN for p in rel.parts):raise Blocked('Forbidden data tree inside output')
        if path.is_file():
            if path.suffix not in ('.json','.jsonl','.log','.txt'):
                # Open writer temporary/lock files are not stable artifacts.
                if '.pending-' in path.name or path.suffix in ('.tmp','.lock'):continue
                raise Blocked('Unexpected output artifact type: '+str(rel))
            files.append(checked_under(path,root))
    return files


def registered_jobs(registration,expected_count=EXPECTED_RECORDS):
    registration=Path(registration).resolve();root=registration.parent;r=read(registration)
    jobs={};plans=[]
    for item in r['plans']:
        path=checked_under(item['path'],root)
        if sha_file(path)!=item['sha256']:raise Blocked('Registered plan hash changed')
        plan=read(path)
        if len(plan['jobs'])!=item['records']:raise Blocked('Registered plan count changed')
        for job in plan['jobs']:
            key=job['job_key']
            if key in jobs:raise Blocked('Duplicate registered job')
            if job['stage'] not in ('main','controls'):raise Blocked('Non-formal stage in export plan')
            jobs[key]=job
        plans.append((path,plan))
    if len(jobs)!=expected_count or r['inventory']['planned_records']!=expected_count:
        raise Blocked('Full main+controls registration must contain 21900 jobs')
    return jobs,plans


def verify_results(formal_out,registration,expected_count=EXPECTED_RECORDS):
    jobs,plans=registered_jobs(registration,expected_count);seen={};invalid=[]
    for path in sorted(Path(formal_out).rglob('result.json')):
        checked_under(path,formal_out)
        try:
            r=read(path);key=r['job_key']
            if key in seen:raise Blocked('Duplicate completed job')
            if key not in jobs:raise Blocked('Unregistered completed job')
            if path.parent.name!=key:raise Blocked('Result job directory mismatch')
            expected=jobs[key]
            if any(r.get(k)!=v for k,v in expected.items()):raise Blocked('Result differs from registered job')
            if r.get('record_complete') is not True:raise Blocked('Incomplete result')
            binding=read(path.parent/'binding.json')
            if binding.get('job')!=expected or r.get('binding_sha256')!=canonical(binding):raise Blocked('Result binding changed')
            ids=r['generated_token_ids'];n=r['output_tokens'];cap=expected['generation']['max_new_tokens']
            if not isinstance(ids,list) or type(n)!=int or n!=len(ids) or not 0<n<=cap or any(type(t)!=int or t<0 for t in ids):raise Blocked('Invalid original tokens')
            if not isinstance(r.get('output_text'),str):raise Blocked('Full original output text absent')
            if r.get('output_cap_reached')!=(n==cap):raise Blocked('Invalid cap flag')
            if r.get('eos_terminated') is not True and n!=cap:raise Blocked('Unexpected termination')
            seen[key]=str(path)
        except (KeyError,ValueError,TypeError,OSError,Blocked) as exc:
            invalid.append({'path':str(path),'error_type':type(exc).__name__,'reason':str(exc)[:300]})
    completion=Path(formal_out)/'completion.json';terminal=False
    if completion.exists():
        c=read(completion)
        terminal=(c.get('stage')=='generation_complete' and c.get('exit_code')==0)
    coordinator_binding=Path(formal_out)/'binding.json'
    binding_ok=coordinator_binding.exists() and read(coordinator_binding).get('registration_sha256')==sha_file(registration)
    passed=len(seen)==expected_count and not invalid and terminal and binding_ok
    return dict(verified_complete=passed,expected_records=expected_count,valid_results=len(seen),
        missing_job_keys=sorted(set(jobs)-set(seen)),invalid_results=invalid,
        coordinator_success=terminal,coordinator_binding_valid=binding_ok),plans


class DigestReader:
    def __init__(self,handle):self.handle=handle;self.hash=hashlib.sha256();self.count=0
    def read(self,size=-1):
        value=self.handle.read(size);self.hash.update(value);self.count+=len(value);return value


def export(folder,registration,output,archive_dir,snapshot=False):
    formal_out=output;out=archive_dir
    folder=Path(folder).resolve();formal_out=checked_under(formal_out,folder)
    registration=checked_under(registration,folder);out=Path(out).resolve()
    if out.exists():raise Blocked('Export directory already exists; preserve previous snapshot')
    if out.is_relative_to(formal_out) or out.is_relative_to(registration.parent):raise Blocked('Export cannot be nested inside source artifacts')
    lock=None
    try:
        if not snapshot:
            import fcntl
            lock=(folder/'production-coordinator.lock').open('a')
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        verification,plans=verify_results(formal_out,registration)
        if not snapshot and not verification['verified_complete']:raise Blocked('Formal run is incomplete or failed validation; use explicit --snapshot')
        paths={}
        def add(path,name):
            path=Path(path)
            if not path.is_file() or path.is_symlink():raise Blocked('Missing/nonregular export artifact')
            if name in paths and paths[name]!=path:raise Blocked('Archive name collision')
            paths[name]=path
        for path in artifact_tree(formal_out):add(path,'formal/'+path.relative_to(formal_out).as_posix())
        for path in artifact_tree(registration.parent):add(path,'registration/'+path.relative_to(registration.parent).as_posix())
        for relative in sorted(PROVENANCE_NAMES):
            path=folder/relative
            if path.exists():add(checked_under(path,folder),'provenance/'+relative)
        for plan_path,plan in plans:
            for field in sorted(PLAN_REFERENCES):
                if field in plan:
                    raw_path=Path(plan[field]);path=checked_under(raw_path,folder.parent.parent)
                    if any(p in FORBIDDEN for p in path.parts) or path.suffix not in ('.json','.jsonl'):raise Blocked('Invalid provenance reference')
                    # References to deployed source/config files may be outside run;
                    # only the explicit JSON file is copied, never its directory.
                    add(path,'provenance/plan-references/'+digest(str(path))[:16]+'-'+path.name)
                    if field=='execution_acceptance_path':
                        for item in read(path).get('evidence',[]):
                            evidence=checked_under(item['path'],folder)
                            if evidence.suffix not in ('.json','.jsonl') or sha_file(evidence)!=item['sha256']:
                                raise Blocked('Execution acceptance evidence changed')
                            add(evidence,'provenance/execution-evidence/'+digest(str(evidence))[:16]+'-'+evidence.name)
        out.mkdir(parents=True);partial=out/'results.tar.gz.partial';entries=[];unstable=[]
        with tarfile.open(partial,'w:gz',compresslevel=1) as tar:
            for name,path in sorted(paths.items()):
                with path.open('rb') as handle:
                    before=os.fstat(handle.fileno());reader=DigestReader(handle)
                    info=tarfile.TarInfo(name);info.size=before.st_size;info.mode=0o600;info.mtime=before.st_mtime
                    tar.addfile(info,reader);after=os.fstat(handle.fileno())
                    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):unstable.append(name)
                    entries.append({'path':name,'source_path':str(path),'size':reader.count,'sha256':reader.hash.hexdigest()})
            if unstable and not snapshot:raise Blocked('Artifact changed during completed export')
            manifest=dict(format='r3-formal-export-v1',created_unix=time.time(),snapshot=snapshot,
                complete=not snapshot and verification['verified_complete'] and not unstable,
                verification=verification,unstable_files=unstable,files=entries,
                excluded=['models','original images','datasets','vendor','full environment'],
                parsing_or_scoring_performed=False)
            payload=(json.dumps(manifest,ensure_ascii=False,sort_keys=True)+'\n').encode()
            info=tarfile.TarInfo('MANIFEST.json');info.size=len(payload);info.mode=0o600;tar.addfile(info,io.BytesIO(payload))
        archive=out/'results.tar.gz';os.replace(partial,archive)
        (out/'MANIFEST.json').write_bytes(payload)
        receipt=dict(format='r3-formal-export-v1',archive_name='results.tar.gz',archive_size=archive.stat().st_size,
            archive_sha256=sha_file(archive),archive_path=str(archive),manifest_path=str(out/'MANIFEST.json'),
            receipt_path=str(out/'export_receipt.json'),manifest_sha256=digest(payload),complete=manifest['complete'],snapshot=snapshot,
            verified_results=verification['valid_results'],expected_records=EXPECTED_RECORDS,files=len(entries))
        write(out/'export_receipt.json',receipt);return receipt
    finally:
        if lock:lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--formal-out',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--snapshot',action='store_true');a=p.parse_args()
    print(json.dumps(export(a.folder,a.registration,a.formal_out,a.out,a.snapshot)))

if __name__=='__main__':main()
