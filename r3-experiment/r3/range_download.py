"""Bounded official HTTP range transfer with immutable verified chunk receipts."""
import hashlib,re,shutil,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from .common import read,write,Blocked


def validate_range(headers,start,end,total,validator):
    if headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':raise Blocked('HTTP range differs')
    if headers.get(validator[0])!=validator[1]:raise Blocked('Remote archive changed during transfer')


def transfer(url,target,progress):
    import requests
    target=Path(target);parts=target.with_name(target.name+'.ranges');parts.mkdir(exist_ok=True)
    with requests.get(url,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'},stream=True,timeout=(20,90)) as r:
        if r.status_code!=206:raise Blocked('Official server must support validated byte ranges')
        match=re.fullmatch(r'bytes 0-0/(\d+)',r.headers.get('Content-Range',''))
        if not match:raise Blocked('Invalid archive size probe')
        total=int(match[1]);etag=r.headers.get('ETag')
        validator=('ETag',etag) if etag and not etag.startswith('W/') else ('Last-Modified',r.headers.get('Last-Modified'))
        if not validator[1]:raise Blocked('No archive identity validator')
    origin=dict(url=url,total=total,validator=list(validator),chunk_bytes=8*1024**2)
    lock=parts/'source.json'
    if lock.exists() and read(lock)!=origin:raise Blocked('Changed ranged source lock')
    if not lock.exists():write(lock,origin)
    # Keep original failed partials and chunks; reserve archive assembly too.
    if shutil.disk_usage(parts).free<2*total+5*1024**3:raise Blocked('Insufficient space for retained chunks and archive')
    def one(start):
        end=min(total-1,start+origin['chunk_bytes']-1);p=parts/(str(start)+'.bin');receipt=p.with_suffix('.json')
        if p.exists():
            if not receipt.exists():raise Blocked('Chunk exists without verification receipt')
            info=read(receipt);data=p.read_bytes()
            if len(data)!=end-start+1 or hashlib.sha256(data).hexdigest()!=info['sha256']:raise Blocked('Corrupt stored chunk')
            return len(data)
        with requests.get(url,headers={'Range':f'bytes={start}-{end}','If-Range':validator[1],'Accept-Encoding':'identity'},timeout=(20,120)) as r:
            if r.status_code!=206:raise Blocked('Unhonored range')
            validate_range(r.headers,start,end,total,validator);data=r.content
        if len(data)!=end-start+1:raise Blocked('Short chunk')
        temp=p.with_name(p.name+'.partial-'+str(time.time_ns()));temp.write_bytes(data);temp.rename(p)
        write(receipt,dict(start=start,end=end,sha256=hashlib.sha256(data).hexdigest()))
        return len(data)
    completed=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(one,start) for start in range(0,total,origin['chunk_bytes'])]
        try:
            for future in as_completed(pending):
                completed+=future.result();progress(bytes=completed,expected_bytes=total)
        except BaseException:
            for future in pending:future.cancel()
            raise
    h=hashlib.sha256();assembly=target.with_name(target.name+'.assembly-'+str(time.time_ns()))
    with assembly.open('xb') as stream:
        for start in range(0,total,origin['chunk_bytes']):
            data=(parts/(str(start)+'.bin')).read_bytes();h.update(data);stream.write(data)
    if target.exists():raise Blocked('Refuse archive overwrite')
    assembly.rename(target)
    return dict(url=url,sha256=h.hexdigest(),bytes=total,validator=list(validator),transport='four_parallel_verified_ranges')
