"""Native LLaVA pad/336/tokenizer input audit, no model answers or labels."""
import argparse,importlib,json,os,time
from pathlib import Path
from .common import read,write,digest,Blocked


def audit(folder,candidates,out):
    if out.exists():raise Blocked('Choose a new immutable audit output')
    from .native_smoke import activate
    activate(folder,'Full')
    import torch
    torch.set_num_threads(4)
    from PIL import Image
    from transformers import AutoTokenizer,CLIPImageProcessor,CLIPVisionConfig
    from llava.model.language_model.llava_llama import LlavaConfig
    from llava.mm_utils import process_images,tokenizer_image_token
    from llava.constants import DEFAULT_IMAGE_TOKEN,IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from .audit_qwen_inputs import question_text
    lock=read(folder/'manifests/model_weights_lock.json')
    tokenizer=AutoTokenizer.from_pretrained(lock['llava7']['path'],local_files_only=True,use_fast=False)
    cfg=LlavaConfig.from_pretrained(lock['llava7']['path'],local_files_only=True);cfg.image_aspect_ratio='pad'
    processor=CLIPImageProcessor.from_pretrained(lock['clip336']['path'],local_files_only=True)
    vision=CLIPVisionConfig.from_pretrained(lock['clip336']['path'],local_files_only=True)
    if vision.image_size!=336 or vision.patch_size!=14 or cfg.mm_vision_select_feature!='patch':raise Blocked('Unexpected native vision policy')
    processor_sha=digest(json.dumps(processor.to_dict(),sort_keys=True))
    tokenizer_sha=digest(json.dumps(tokenizer.get_vocab(),sort_keys=True))
    suffix=read(Path(__file__).resolve().parent.parent/'round3_spec.json')['prompt']['suffix']
    rows=[json.loads(x) for x in candidates.read_text(encoding='utf-8').splitlines()]
    if len({r['uid'] for r in rows})!=len(rows):raise Blocked('Duplicate candidate uid')
    state=dict(pid=os.getpid(),stage='audit',total=len(rows),completed=0,production_started=False,
               candidate_sha256=digest(candidates.read_bytes()))
    def update(**kw):
        state.update(kw,updated_at=time.time());write(out.with_suffix('.status.json'),state)
    out.parent.mkdir(parents=True,exist_ok=True)
    try:
        update()
        with out.open('x',encoding='utf-8') as f:
            for i,row in enumerate(rows):
                path=Path(row['image_path'])
                if digest(path.read_bytes())!=row['image_sha256']:raise Blocked('Source image changed')
                conv=conv_templates['llava_v1'].copy()
                conv.append_message(conv.roles[0],DEFAULT_IMAGE_TOKEN+'\n'+question_text(row,suffix));conv.append_message(conv.roles[1],None)
                prompt=conv.get_prompt();ids=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt')
                if int((ids==IMAGE_TOKEN_INDEX).sum())!=1:raise Blocked('Expected exactly one image token')
                with Image.open(path) as im:pixels=process_images([im.convert('RGB')],processor,cfg)
                if not isinstance(pixels,torch.Tensor) or list(pixels.shape)!=[1,3,336,336]:raise Blocked('Unexpected native image tensor')
                expanded=len(ids)-1+576
                entry=dict(eligible=expanded<=2048,expanded_prompt_tokens=expanded,visual_tokens=576,
                    processed_size=[336,336],native_context=4096,processor_sha256=processor_sha,
                    tokenizer_sha256=tokenizer_sha,prompt_sha256=digest(json.dumps(ids.tolist())))
                f.write(json.dumps(dict(uid=row['uid'],model='llava7',audit=entry))+'\n')
                if (i+1)%20==0:f.flush();update(completed=i+1)
        update(stage='complete',completed=len(rows),exit_code=0,audit_sha256=digest(out.read_bytes()))
    except BaseException as exc:
        update(stage='failed',exit_code=1,error_type=type(exc).__name__);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',required=True)
    p.add_argument('--candidates',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();audit(Path(a.folder),Path(a.candidates),Path(a.out))
