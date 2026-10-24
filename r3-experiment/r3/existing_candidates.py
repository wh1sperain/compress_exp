"""Extract frozen Doc/Info pools and existing MathVista bytes, without resampling."""
import argparse,json
from pathlib import Path
from .common import read,write,digest,Blocked
from .artifacts import sha256


def prepare(folder,base):
    import pyarrow.parquet as pq
    from PIL import Image
    out=folder/'data/existing-pools-v1';out.mkdir(parents=True,exist_ok=True)
    target=out/'candidates_unvalidated.jsonl'
    if target.exists():raise Blocked('Existing candidates remain immutable')
    old=base/'visual_reasoning_v1';manifest=read(old/'manifest.json')
    meta={r['uid']:r for r in read(old/'all_metadata.json')}
    rows=[];sources={};groups={}
    for task in ['DocVQA','InfographicVQA']:
        uids=manifest['samples'][task]['formal']
        if len(uids)!=500 or len(set(uids))!=500:raise Blocked('Expected original frozen 500-question pool')
        for uid in uids:
            r=meta[uid];groups.setdefault(r['source_file'],[]).append(r)
    def image(data):
        if not data:raise Blocked('No embedded source image bytes')
        sha=digest(data);p=out/'images'/sha;p.parent.mkdir(exist_ok=True)
        if p.exists() and digest(p.read_bytes())!=sha:raise Blocked('Existing image differs')
        if not p.exists():p.write_bytes(data)
        with Image.open(p) as im:im.convert('RGB').load();size=list(im.size)
        return dict(image_path=str(p),image_sha256=sha,original_size=size,single_image=True,rgb_decodable=True,public_label=True,native_inputs={})
    for path,items in sorted(groups.items()):
        source=Path(path);h=sha256(source);sources[path]=h;table=pq.read_table(source)
        for r in items:
            q=table.slice(r['row_index'],1).to_pylist()[0]
            if str(q['questionId'])!=r['question_id'] or q['question']!=r['question'] or q['answers']!=r['answers']:
                raise Blocked('Frozen question/source row mismatch')
            rows.append(dict(uid=r['uid'],task=r['dataset'],question=r['question'],answers=r['answers'],
                source_revision='sha256:'+h,source_file=path,**image(q['image']['bytes'])))
    files=list((base/'MathVista/data').glob('testmini-*.parquet'))
    if len(files)!=1:raise Blocked('Ambiguous MathVista source')
    h=sha256(files[0]);sources[str(files[0])]=h
    for batch in pq.ParquetFile(files[0]).iter_batches(batch_size=32):
        for q in batch.to_pylist():
            if q['metadata']['split']!='testmini':raise Blocked('Wrong MathVista split')
            rows.append(dict(uid='MathVista-testmini:'+q['pid'],task='MathVista-testmini',question=q['question'],
                answer=q['answer'],options=q['choices'] or [],answer_type=q['answer_type'],precision=q['precision'],
                source_revision='sha256:'+h,source_file=str(files[0]),**image(q['decoded_image']['bytes'])))
    if len(rows)!=2000 or len({r['uid'] for r in rows})!=2000:raise Blocked('Unexpected candidate count')
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    target.write_text(payload,encoding='utf-8')
    write(out/'source_receipt.json',dict(candidate_sha256=digest(payload),sources=sources,
        old_manifest_sha256=digest((old/'manifest.json').read_bytes()),count=len(rows),native_inputs_validated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--base',required=True)
    a=p.parse_args();prepare(Path(a.folder),Path(a.base))
