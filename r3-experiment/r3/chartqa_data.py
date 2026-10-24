"""Acquire official ChartQA human-test inputs; no sampling or generation."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import quote
from .common import read,write,Blocked,digest


def verify_blob(data,sha):
    actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if actual!=sha:raise Blocked('Official Git blob hash mismatch')


def acquire(folder):
    import fcntl
    import requests
    from PIL import Image
    lock=(folder/'chartqa-prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=folder/'telemetry'/('chartqa-'+str(time.time_ns())+'.json')
    state=dict(pid=os.getpid(),stage='resolve_source',production_started=False,updated_at=time.time())
    status=folder/'telemetry/chartqa_state.json'
    def update(**kw):
        state.update(kw,updated_at=time.time());write(status,state);write(attempt,state)
    session=requests.Session()
    def get(url):
        response=session.get(url,timeout=(15,90));response.raise_for_status();return response
    try:
        update()
        source=folder/'manifests/chartqa_source.json'
        api='https://api.github.com/repos/vis-nlp/ChartQA'
        if source.exists():origin=read(source)
        else:
            commit=get(api+'/commits/main').json()['sha']
            tree=get(api+'/git/trees/'+commit+'?recursive=1').json()
            if tree.get('truncated'):raise Blocked('Incomplete source tree')
            entries={e['path']:e for e in tree['tree'] if e['type']=='blob' and e['path'].startswith('ChartQA Dataset/test/')}
            origin=dict(repository='https://github.com/vis-nlp/ChartQA',revision=commit,files=entries)
            write(source,origin)
        root=folder/'data/ChartQA'/origin['revision'];root.mkdir(parents=True,exist_ok=True)
        def fetch(relative):
            entry=origin['files'][relative]
            target=root/Path(relative).relative_to('ChartQA Dataset/test')
            if target.exists():data=target.read_bytes();verify_blob(data,entry['sha'])
            else:
                data=get('https://raw.githubusercontent.com/vis-nlp/ChartQA/'+origin['revision']+'/'+quote(relative)).content
                verify_blob(data,entry['sha']);target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:stream.write(data)
            return target,data
        _,raw=fetch('ChartQA Dataset/test/test_human.json')
        questions=json.loads(raw);rows=[];images={}
        update(stage='download_human_test_images',questions=len(questions),completed=0)
        for i,q in enumerate(questions):
            name=q['imgname']
            if Path(name).name!=name:raise Blocked('Unsafe image name')
            if name not in images:
                path,data=fetch('ChartQA Dataset/test/png/'+name)
                with Image.open(path) as image:
                    image.convert('RGB').load();size=list(image.size)
                images[name]=(str(path),digest(data),size)
            path,sha,size=images[name]
            rows.append(dict(uid='ChartQA-human-test:'+str(i),task='ChartQA-human-test',question=q['query'],
                answer=q['label'],image_path=path,image_sha256=sha,original_size=size,
                source_revision=origin['revision'],source_file='test_human.json',
                public_label=True,single_image=True,rgb_decodable=True,native_inputs={}))
            if (i+1)%20==0:update(completed=i+1,images=len(images))
        target=root/'candidates_unvalidated.jsonl'
        payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
        if target.exists() and target.read_text(encoding='utf-8')!=payload:raise Blocked('Existing candidates differ')
        if not target.exists():target.write_text(payload,encoding='utf-8')
        update(stage='source_ready_input_audit_required',exit_code=0,completed=len(rows),images=len(images),
               candidates=str(target),candidate_sha256=digest(payload),native_inputs_validated=False)
    except BaseException as exc:
        update(stage='failed',exit_code=1,error_type=type(exc).__name__)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',required=True)
    a=p.parse_args();acquire(Path(a.folder))
