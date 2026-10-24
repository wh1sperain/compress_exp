"""Local-only resumable formal artifact download using existing connection(3).

No remote command is executed, no credentials copied, no models/images fetched.
Archive receipt, archive SHA256 and per-file manifest are verified. Extraction is
optional and accepts only regular allowlisted paths, never links or traversal.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import sys
import tarfile
import time


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe_member(name):
    if '\\' in name or ':' in name or name.startswith('/') or '\x00' in name:raise ValueError('Unsafe archive name')
    parts=name.split('/')
    if any(p in ('','.','..') for p in parts):raise ValueError('Unsafe archive path')
    if name!='MANIFEST.json' and parts[0] not in ('formal','registration','provenance'):raise ValueError('Unexpected artifact namespace')
    return parts


def checked_write(path,data):
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Immutable local metadata changed')
        return
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def download(sftp,remote_path,target,expected_size,expected_sha):
    target=Path(target);partial=target.with_name(target.name+'.partial')
    if target.exists():
        if target.stat().st_size!=expected_size or sha_file(target)!=expected_sha:raise ValueError('Existing archive differs')
        return
    if sftp.stat(remote_path).st_size!=expected_size:raise ValueError('Remote archive size changed')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>expected_size:raise ValueError('Partial larger than archive; original retained')
    # Compare partial prefix to server before appending; do not continue a corrupt
    # or different archive's partial bytes and then silently replace them.
    with sftp.file(remote_path,'rb') as source:
        if offset:
            with partial.open('rb') as local:
                remaining=offset
                while remaining:
                    block=local.read(min(1024*1024,remaining));remote=source.read(len(block))
                    if remote!=block:raise ValueError('Partial prefix mismatch; original retained')
                    remaining-=len(block)
        with partial.open('ab') as dest:
            remaining=expected_size-offset
            while remaining:
                chunk=source.read(min(1024*1024,remaining))
                if not chunk:raise IOError('Interrupted archive download')
                dest.write(chunk);remaining-=len(chunk)
            dest.flush();os.fsync(dest.fileno())
    if partial.stat().st_size!=expected_size or sha_file(partial)!=expected_sha:raise ValueError('Downloaded archive checksum mismatch; partial retained')
    os.replace(partial,target)


def verify_archive(archive,manifest,extract_to=None):
    expected={}
    for item in manifest['files']:
        safe_member(item['path'])
        if item['path'] in expected:raise ValueError('Duplicate manifest path')
        expected[item['path']]=item
    seen=set();manifest_found=False
    staging=None
    if extract_to is not None:
        extract_to=Path(extract_to)
        if extract_to.exists():raise ValueError('Extraction target exists')
        staging=extract_to.with_name(extract_to.name+'.extracting-'+str(time.time_ns()));staging.mkdir(parents=True)
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            parts=safe_member(member.name)
            if not member.isfile() or member.issym() or member.islnk():raise ValueError('Only regular archive members accepted')
            if member.name in seen:raise ValueError('Duplicate archive path')
            seen.add(member.name)
            if member.name=='MANIFEST.json':
                data=tar.extractfile(member).read()
                if json.loads(data)!=manifest:raise ValueError('Embedded manifest differs')
                manifest_found=True;continue
            if member.name not in expected or member.size!=expected[member.name]['size']:raise ValueError('Unlisted or wrong-size archive member')
            h=hashlib.sha256();total=0;dest=None
            if staging is not None:
                path=staging.joinpath(*parts)
                if not path.resolve().is_relative_to(staging.resolve()):raise ValueError('Extraction escape')
                path.parent.mkdir(parents=True,exist_ok=True);dest=path.open('xb')
            try:
                with tar.extractfile(member) as source:
                    while True:
                        data=source.read(1024*1024)
                        if not data:break
                        h.update(data);total+=len(data)
                        if dest:dest.write(data)
            finally:
                if dest:dest.close()
            if total!=member.size or h.hexdigest()!=expected[member.name]['sha256']:raise ValueError('Artifact checksum differs')
    if not manifest_found or seen!=set(expected)|{'MANIFEST.json'}:raise ValueError('Incomplete archive')
    if staging is not None:
        (staging/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(staging,extract_to)
    return {'verified_files':len(expected),'complete':manifest['complete'],'snapshot':manifest['snapshot']}


def collect(sftp,receipt_remote,out,extract=False,lifecycle_remote=None):
    receipt_remote=PurePosixPath(receipt_remote)
    if receipt_remote.name!='export_receipt.json' or '..' in receipt_remote.parts:raise ValueError('Explicit export receipt required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with sftp.file(str(receipt_remote),'rb') as f:receipt_bytes=f.read()
    if len(receipt_bytes)>1024*1024:raise ValueError('Unexpectedly large receipt')
    receipt=json.loads(receipt_bytes)
    if receipt.get('format')!='r3-formal-export-v1' or receipt.get('archive_name')!='results.tar.gz':raise ValueError('Unsupported export receipt')
    checked_write(out/'export_receipt.json',receipt_bytes)
    manifest_remote=str(receipt_remote.parent/'MANIFEST.json')
    with sftp.file(manifest_remote,'rb') as f:manifest_bytes=f.read()
    if hashlib.sha256(manifest_bytes).hexdigest()!=receipt['manifest_sha256']:raise ValueError('Remote manifest checksum mismatch')
    manifest=json.loads(manifest_bytes)
    if manifest['complete']!=receipt['complete'] or manifest['snapshot']!=receipt['snapshot']:raise ValueError('Manifest/receipt state mismatch')
    checked_write(out/'MANIFEST.json',manifest_bytes)
    archive=out/'results.tar.gz'
    download(sftp,str(receipt_remote.parent/receipt['archive_name']),archive,receipt['archive_size'],receipt['archive_sha256'])
    verification=verify_archive(archive,manifest,out/'contents' if extract else None)
    # Final lifecycle is intentionally outside the tar: it is written after
    # export. Capture separately without presenting an earlier exporting state
    # as final lifecycle completion.
    if lifecycle_remote:
        lifecycle_path=PurePosixPath(lifecycle_remote)
        if lifecycle_path.name!='lifecycle_completion.json' or '..' in lifecycle_path.parts:raise ValueError('Expected final lifecycle receipt')
        with sftp.file(str(lifecycle_path),'rb') as f:lifecycle=f.read()
        json.loads(lifecycle)
        checked_write(out/'lifecycle_completion.json',lifecycle)
        verification['lifecycle_sha256']=hashlib.sha256(lifecycle).hexdigest()
    verification.update(archive_sha256=receipt['archive_sha256'],verified_at=time.time(),parsed_or_scored=False)
    (out/'local_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    return verification


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--extract',action='store_true');p.add_argument('--lifecycle-receipt');a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'Compress/tools'))
    import remote
    client=None
    try:
        client=remote.connection(3)
        with client.open_sftp() as sftp:result=collect(sftp,a.receipt,a.out,a.extract,a.lifecycle_receipt)
        print(json.dumps(result));return 0
    except Exception as exc:
        # Never print exception text containing host, credentials or signed URLs.
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'partial_files_preserved':True}));return 1
    finally:
        if client:client.close()

if __name__=='__main__':raise SystemExit(main())
