"""Official GQA v1.2 balanced validation acquisition; no sample selection."""
import argparse,hashlib,json,os,time,zipfile
from pathlib import Path
from .common import read,write,Blocked,digest


def acquire(folder):
    import fcntl,requests
    from PIL import Image
    lock=(folder/'gqa-prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=folder/'data/GQA-v1.2';root.mkdir(parents=True,exist_ok=True)
    attempt=folder/'telemetry'/('gqa-'+str(time.time_ns())+'.json')
    state=dict(pid=os.getpid(),production_started=False,stage='download',updated_at=time.time())
    def update(**kw):
        state.update(kw,updated_at=time.time());write(folder/'telemetry/gqa_state.json',state);write(attempt,state)
    def download(name):
        url='https://downloads.cs.stanford.edu/nlp/data/gqa/'+name
        target=root/name;receipt=root/(name+'.receipt.json')
        if target.exists():
            if not receipt.exists():raise Blocked('Archive exists without receipt')
            expected=read(receipt)['sha256'];h=hashlib.sha256()
            with target.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            if h.hexdigest()!=expected:raise Blocked('Existing archive hash mismatch')
            return target
        update(stage='download',file=name,bytes=0)
        from .range_download import transfer
        info=transfer(url,target,update)
        if not zipfile.is_zipfile(target):raise Blocked('Official response is not ZIP')
        write(receipt,dict(info,source_version='GQA-v1.2'))
        return target
    try:
        update()
        questions=download('questions1.2.zip')
        with zipfile.ZipFile(questions) as z:
            names=[n for n in z.namelist() if Path(n).name=='val_balanced_questions.json']
            if len(names)!=1:raise Blocked('Expected unique official balanced validation member')
            raw=z.read(names[0]);data=json.loads(raw)
        if any(q.get('isBalanced') is False for q in data.values()):raise Blocked('Unexpected unbalanced question')
        image_ids={str(q['imageId']) for q in data.values()}
        if any(not i.isalnum() for i in image_ids):raise Blocked('Unsafe image ID')
        archive=download('images.zip');image_info={}
        update(stage='extract_validation_images',images_expected=len(image_ids),images_completed=0)
        with zipfile.ZipFile(archive) as z:
            members={}
            for name in z.namelist():
                p=Path(name)
                if p.suffix.lower()=='.jpg' and p.stem in image_ids:
                    if p.stem in members:raise Blocked('Duplicate image ID in archive')
                    members[p.stem]=name
            if set(members)!=image_ids:raise Blocked('Missing validation images')
            for i,(identity,name) in enumerate(sorted(members.items())):
                payload=z.read(name);path=root/'val_images'/(identity+'.jpg');path.parent.mkdir(exist_ok=True)
                if path.exists() and digest(path.read_bytes())!=digest(payload):raise Blocked('Existing image differs')
                if not path.exists():path.write_bytes(payload)
                with Image.open(path) as im:im.convert('RGB').load();size=list(im.size)
                image_info[identity]=(str(path),digest(payload),size)
                if i%100==0:update(images_completed=i+1)
        revision='sha256:'+read(root/'questions1.2.zip.receipt.json')['sha256'];rows=[]
        for uid,q in sorted(data.items()):
            path,sha,size=image_info[str(q['imageId'])]
            rows.append(dict(uid='GQA-balanced-val:'+uid,task='GQA-balanced-val',question=q['question'],answer=q['answer'],
                image_path=path,image_sha256=sha,original_size=size,source_revision=revision,
                source_file=names[0],single_image=True,rgb_decodable=True,public_label=True,native_inputs={}))
        payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);target=root/'candidates_unvalidated.jsonl'
        if target.exists():raise Blocked('Candidate output already exists')
        target.write_text(payload,encoding='utf-8')
        update(stage='source_ready_input_audit_required',exit_code=0,questions=len(rows),images=len(image_info),
               candidates=str(target),candidate_sha256=digest(payload),native_inputs_validated=False)
    except BaseException as exc:
        update(stage='failed',exit_code=1,error_type=type(exc).__name__);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',required=True)
    a=p.parse_args();acquire(Path(a.folder))
