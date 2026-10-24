"""Audit real Mini compressor and prefusion tensors, without answer generation."""
import argparse,json,os,time
from pathlib import Path
from .common import read,write,digest,Blocked


def audit(folder,candidates,out):
    if out.exists():raise Blocked('Choose new audit attempt')
    from .native_smoke import activate
    activate(folder,'MiniNative')
    import torch
    torch.set_num_threads(4)
    from PIL import Image
    from transformers import AutoTokenizer
    from llavamini.model.language_model.llavamini_llama import LlavaMiniConfig,LlavaMiniLlamaForCausalLM
    from llavamini.mm_utils import process_images,tokenizer_image_token
    from llavamini.constants import DEFAULT_IMAGE_TOKEN,IMAGE_TOKEN_INDEX
    from llavamini.conversation import conv_templates
    from .audit_qwen_inputs import question_text
    lock=read(folder/'manifests/model_weights_lock.json');cfg=LlavaMiniConfig.from_pretrained(lock['mini8']['path'],local_files_only=True)
    cfg.mm_vision_tower=lock['clip336']['path'];cfg.unfreeze_mm_vision_tower=True
    model,info=LlavaMiniLlamaForCausalLM.from_pretrained(lock['mini8']['path'],config=cfg,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',local_files_only=True,output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise Blocked('Mini strict loading failed')
    model.eval().requires_grad_(False);model.to(dtype=torch.bfloat16)
    if cfg.resolution_ratio!=1:raise Blocked('Need separate multi-crop audit')
    tokenizer=AutoTokenizer.from_pretrained(lock['mini8']['path'],use_fast=False,local_files_only=True)
    processor=model.get_vision_tower().image_processor
    processor_sha=digest(json.dumps(processor.to_dict(),sort_keys=True))
    window=min(cfg.max_position_embeddings,getattr(cfg,'tokenizer_model_max_length',cfg.max_position_embeddings))
    seen={}
    def compressor_hook(module,args,output):seen['visual_tokens']=int(output[0].shape[1])
    handles=[model.model.compressor.register_forward_hook(compressor_hook)]
    def prehook(index):
        def hook(module,args,kwargs):
            hidden=args[0] if args else kwargs['hidden_states'];seen.setdefault('prefusion',{})[str(index)]=int(hidden.shape[1])
        return hook
    for i,layer in enumerate(model.model.prefusion_layers):handles.append(layer.register_forward_pre_hook(prehook(i),with_kwargs=True))
    rows=[]
    for source in candidates:rows.extend(json.loads(x) for x in source.read_text(encoding='utf-8').splitlines())
    if len({r['uid'] for r in rows})!=len(rows):raise Blocked('Duplicate candidate UID')
    suffix=read(Path(__file__).resolve().parent.parent/'round3_spec.json')['prompt']['suffix']
    state=dict(pid=os.getpid(),stage='audit',completed=0,total=len(rows),production_started=False)
    def update(**kw):
        state.update(kw,updated_at=time.time());write(out.with_suffix('.status.json'),state)
    out.parent.mkdir(parents=True,exist_ok=True)
    try:
        update()
        with out.open('x',encoding='utf-8') as f,torch.inference_mode():
            for i,row in enumerate(rows):
                path=Path(row['image_path'])
                if digest(path.read_bytes())!=row['image_sha256']:raise Blocked('Image changed')
                conv=conv_templates['llava_llama_3_1'].copy();conv.append_message(conv.roles[0],DEFAULT_IMAGE_TOKEN+'\n'+question_text(row,suffix));conv.append_message(conv.roles[1],None)
                prompt=conv.get_prompt();ids=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()
                if int((ids==IMAGE_TOKEN_INDEX).sum())!=1:raise Blocked('Expected one image')
                # Refuse native truncation before auditing actual tensors.
                if ids.shape[1]+576+cfg.compressor_size**2>window:raise Blocked('Prefusion input needs explicit ineligible audit')
                with Image.open(path) as image:pixels=process_images([image.convert('RGB')],processor,cfg)
                if not isinstance(pixels,torch.Tensor):raise Blocked('Expected single tensor path')
                pixels=pixels.unsqueeze(0).to(device='cuda:0',dtype=torch.bfloat16);seen.clear()
                prepared=model.prepare_inputs_labels_for_multimodal(ids,None,None,None,None,pixels)
                embeds=prepared[4];n=int(embeds.shape[1]);visual=seen['visual_tokens']
                if n!=ids.shape[1]-1+visual or len(seen.get('prefusion',{}))!=cfg.prefusion_layer_num:raise Blocked('Mini text/prefusion budget mismatch')
                entry=dict(eligible=n+2048<=window,expanded_prompt_tokens=n,visual_tokens=visual,
                    processed_size=list(pixels.shape[-2:]),native_context=window,processor_sha256=processor_sha,
                    prompt_sha256=digest(json.dumps(ids[0].tolist())),prefusion_lengths=seen['prefusion'],
                    model_revision=lock['mini8']['revision'])
                f.write(json.dumps(dict(uid=row['uid'],model='mini8',audit=entry))+'\n')
                if (i+1)%10==0:f.flush();update(completed=i+1)
        update(stage='complete',completed=len(rows),exit_code=0,audit_sha256=digest(out.read_bytes()))
    except BaseException as exc:
        update(stage='failed',exit_code=1,error_type=type(exc).__name__);raise
    finally:
        for h in handles:h.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--candidates',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();audit(Path(a.folder),[Path(x) for x in a.candidates],Path(a.out))
