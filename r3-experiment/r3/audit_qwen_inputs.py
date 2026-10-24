"""CPU-only native Qwen processor audit; does not generate or score answers."""
import argparse,json,os,time
from pathlib import Path
from .common import read,write,digest,Blocked


def question_text(row,suffix):
    text=row['question']
    options=row.get('options') or []
    if options:
        if not isinstance(options,list) or len(options)>26:raise Blocked('Unsupported option layout')
        text+='\n'+'\n'.join('('+chr(65+i)+') '+str(v) for i,v in enumerate(options))
        text+='\nFor this multiple-choice question, give only the option letter after FINAL ANSWER:.'
    return text+suffix


def audit(folder,candidates,out,model_key="qwen8"):
    if model_key not in ("qwen8","qwen4"):raise Blocked("Unsupported Qwen checkpoint")
    if out.exists():raise Blocked('Input audit output already exists; choose new attempt')
    import torch
    torch.set_num_threads(4)
    from PIL import Image
    from .qwen_smoke import activate
    activate(folder)
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    weights=read(folder/'manifests/model_weights_lock.json')[model_key]
    processor=Qwen3VLProcessor.from_pretrained(weights['path'],local_files_only=True,min_pixels=262144,max_pixels=4194304)
    config=processor.image_processor.to_dict()
    if config['size']!={'shortest_edge':262144,'longest_edge':4194304}:raise Blocked('Pixel policy differs')
    processor_sha=digest(json.dumps(config,sort_keys=True))
    tokenizer_sha=digest(processor.tokenizer.backend_tokenizer.to_str())
    template_sha=digest(processor.chat_template)
    image_id=processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    protocol=read(Path(__file__).resolve().parent.parent/'round3_spec.json')
    rows=[json.loads(line) for line in candidates.read_text(encoding='utf-8').splitlines()]
    if len({r['uid'] for r in rows})!=len(rows):raise Blocked('Duplicate input uid')
    out.parent.mkdir(parents=True,exist_ok=True)
    status=out.with_suffix('.status.json')
    state=dict(pid=os.getpid(),stage='audit',completed=0,total=len(rows),production_started=False,
               candidate_sha256=digest(candidates.read_bytes()))
    def update(**kw):
        state.update(kw,updated_at=time.time());write(status,state)
    try:
        update()
        with out.open('x',encoding='utf-8') as stream:
            for i,row in enumerate(rows):
                path=Path(row['image_path'])
                if digest(path.read_bytes())!=row['image_sha256']:raise Blocked('Image changed after source preparation')
                text=question_text(row,protocol['prompt']['suffix'])
                messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}]
                prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                with Image.open(path) as im:
                    batch=processor(text=[prompt],images=[im.convert('RGB')],return_tensors='pt')
                ids=batch['input_ids'][0];n=int((ids==image_id).sum());grid=batch['image_grid_thw'][0].tolist()
                if n!=grid[0]*grid[1]*grid[2]//config['merge_size']**2:raise Blocked('Native grid/token disagreement')
                entry=dict(eligible=len(ids)<=8192,expanded_prompt_tokens=len(ids),visual_tokens=n,
                    processed_size=[grid[1]*config['patch_size'],grid[2]*config['patch_size']],
                    processor_sha256=processor_sha,tokenizer_sha256=tokenizer_sha,chat_template_sha256=template_sha,
                    prompt_sha256=digest(json.dumps(ids.tolist())),native_context=65536)
                stream.write(json.dumps({'uid':row['uid'],'model':model_key,'audit':entry})+'\n')
                if (i+1)%20==0:stream.flush();update(completed=i+1)
        update(stage='complete',completed=len(rows),exit_code=0,audit_sha256=digest(out.read_bytes()))
    except BaseException as exc:
        update(stage='failed',exit_code=1,error_type=type(exc).__name__);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',required=True)
    p.add_argument('--candidates',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model-key',choices=['qwen8','qwen4'],default='qwen8')
    a=p.parse_args();audit(Path(a.folder),Path(a.candidates),Path(a.out),a.model_key)
