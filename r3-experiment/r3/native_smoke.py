"""Author model loading and synthetic-image BF16 smoke; not research data.

Use a fresh isolated process per author package. Full/native/long trajectory
equivalence must still be established before any production job is admitted.
"""
from __future__ import annotations
import argparse
import importlib
import sys
import time
from pathlib import Path
from .common import read,write,Blocked,digest
from .native_sources import visionzip_arguments


def activate(folder,method):
    sources=read(folder/'manifests/source_lock.json')
    names=['LLaVA-Mini'] if method=='MiniNative' else ['AgilePruner'] if method.startswith('Agile') else ['PriorTR']
    if method.startswith('VisionZip'):names+=['VisionZip']
    if method.startswith('FastV'):names+=['FastV']
    paths={}
    for name in names:
        entry=sources[name];root=Path(entry['path']);paths[name]=root
        for relative,sha in entry['files'].items():
            if relative.endswith('.py') and digest((root/relative).read_bytes())!=sha:
                raise Blocked('Changed author source: '+relative)
    if method=='MiniNative':sys.path.insert(0,str(paths['LLaVA-Mini']))
    elif method.startswith('Agile'):sys.path.insert(0,str(paths['AgilePruner']))
    else:sys.path.insert(0,str(paths['PriorTR']/'image/LLaVA'))
    if method.startswith('VisionZip'):sys.path.insert(0,str(paths['VisionZip']))
    if method.startswith('FastV'):
        sys.path.insert(0,str(paths['FastV']/'src/FastV/llava-hf/transformers/src'))
    return paths


def run(folder,method,out):
    if out.exists() or out.with_suffix('.prefix.json').exists():
        raise Blocked('Existing smoke attempt must remain immutable; choose a new output path')
    paths=activate(folder,method)
    import torch
    from PIL import Image
    from transformers import AutoTokenizer
    lock=read(folder/'manifests/model_weights_lock.json')
    mini=method=='MiniNative';key='mini8' if mini else 'llava7'
    model_path=lock[key]['path'];clip=lock['clip336']['path']
    prefix='llavamini' if mini else 'llava'
    if mini:
        module=importlib.import_module('llavamini.model.language_model.llavamini_llama')
        cls,cfgcls=module.LlavaMiniLlamaForCausalLM,module.LlavaMiniConfig
    else:
        module=importlib.import_module('llava.model.language_model.llava_llama')
        cls,cfgcls=module.LlavaLlamaForCausalLM,module.LlavaConfig
        if method.startswith('PriorTR'):
            cls=importlib.import_module('llava.vtr.model.priortr_llava').PriorTRLlava
    cfg=cfgcls.from_pretrained(model_path,local_files_only=True)
    cfg.mm_vision_tower=clip
    if not mini:cfg.image_aspect_ratio='pad'
    keep=144 if method.endswith('25') else 288 if method.endswith('50') else 576
    loading=dict(config=cfg,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,attn_implementation='eager',local_files_only=True,output_loading_info=True)
    if method.startswith('Agile'):loading['visual_token_num']=keep
    model,loading_info=cls.from_pretrained(model_path,**loading)
    if loading_info.get('missing_keys') or loading_info.get('unexpected_keys') or loading_info.get('mismatched_keys'):
        write(out.with_suffix('.loading-failure.json'),loading_info)
        raise Blocked('Checkpoint tensor loading did not match; see loading-failure receipt')
    model.eval()
    vision=model.get_vision_tower()
    if not vision.is_loaded:vision.load_model(device_map='cuda:0')
    model.to(dtype=torch.bfloat16,device='cuda:0')
    tokenizer=AutoTokenizer.from_pretrained(model_path,use_fast=False,local_files_only=True)
    if len(tokenizer)!=model.get_input_embeddings().weight.shape[0]:
        raise Blocked('Tokenizer/embedding mismatch; do not randomly resize')
    if method.startswith('PriorTR'):
        VTRConfig=importlib.import_module('llava.vtr.config').VTRConfig
        model.setup_vtr(VTRConfig(enabled=True,strategy='priortr',prune_layer=3,keep_tokens=keep,query_aggregation='question',head_aggregation='mean'))
    if method.startswith('VisionZip'):
        from visionzip import visionzip
        model=visionzip(model,**visionzip_arguments(keep))
    mm=importlib.import_module(prefix+'.mm_utils')
    constants=importlib.import_module(prefix+'.constants')
    conversation=importlib.import_module(prefix+'.conversation')
    # The synthetic image and short probe are never written to raw/main.
    image=Image.new('RGB',(640,480),(127,100,80))
    spec=read(Path(__file__).resolve().parent.parent/'round3_spec.json')
    text=constants.DEFAULT_IMAGE_TOKEN+'\nDescribe this image.'+spec['prompt']['suffix']
    conv=conversation.conv_templates['llava_llama_3_1' if mini else 'llava_v1'].copy()
    conv.append_message(conv.roles[0],text);conv.append_message(conv.roles[1],None)
    prompt=conv.get_prompt()
    ids=mm.tokenizer_image_token(prompt,tokenizer,constants.IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()
    pixels=mm.process_images([image],vision.image_processor,model.config)
    if not isinstance(pixels,torch.Tensor):raise Blocked('Expected official single-image tensor path')
    pixels=pixels.to(device='cuda:0',dtype=torch.bfloat16)
    if mini:
        if model.config.resolution_ratio!=1:raise Blocked('Need checkpoint-specific crop audit')
        pixels=pixels.unsqueeze(0)
    if method.startswith('FastV'):
        import functools
        import types
        from .compat import llava_forward_with_cache_position
        model.forward=types.MethodType(llava_forward_with_cache_position,model)
        start=int((ids[0]==constants.IMAGE_TOKEN_INDEX).nonzero()[0])
        if not hasattr(model.model,'fastv_forward'):raise Blocked('Frozen FastV author forward absent')
        if method!='FastVFull':
            model.model.forward=functools.partial(model.model.fastv_forward,fastv_config=dict(fastv_k=2,fastv_r=1-keep/576,image_token_start_index=start,image_token_length=576))
    layer_shapes={};handles=[]
    def hook(index):
        def fn(module,args,kwargs):
            hidden=args[0] if args else kwargs['hidden_states']
            layer_shapes.setdefault(str(index),[]).append(list(hidden.shape))
        return fn
    for i,layer in enumerate(model.model.layers):handles.append(layer.register_forward_pre_hook(hook(i),with_kwargs=True))
    from .reference_stream import generate_native
    job={'job_key':'synthetic-smoke-'+method,'seed':1234,'generation':dict(max_new_tokens=8,temperature=1.,top_k=20,top_p=.95)}
    started=time.time()
    result=generate_native(model,{'inputs':ids,'images':pixels},tokenizer,job,out.with_suffix('.prefix.json'))
    for h in handles:h.remove()
    expected_full=ids.shape[1]-1+576
    for index,shapes in layer_shapes.items():
        expected=expected_full
        if method.startswith(('Agile','VisionZip')):expected=expected_full-576+keep
        if method.startswith('FastV') and int(index)>=2:expected=expected_full-576+keep
        # PriorTR prunes after its 1-based layer 3; record first and assert below.
        if method.startswith('PriorTR') and int(index)>=3:expected=expected_full-576+keep
        if not mini and shapes[0][1]!=expected:raise Blocked('Unexpected layer token budget at '+index)
    write(out,{'smoke_passed':True,'native_gate_passed':False,'production_ready':False,'method':method,
               'synthetic_only':True,'wall_seconds':time.time()-started,'layer_hidden_shapes':layer_shapes,
               'prompt_ids':ids.cpu().tolist(),'prompt_sha256':digest(prompt),'result':result,
               'model_revision':lock[key]['revision'],'clip_revision':lock['clip336']['revision'],
               'weight_dtype':str(next(model.parameters()).dtype),'loading_info':loading_info})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',required=True);p.add_argument('--method',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(Path(a.folder),a.method,Path(a.out))


if __name__=='__main__':main()
