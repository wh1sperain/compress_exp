"""Frozen author Qwen3 native path; synthetic smoke only, never a gate."""
import argparse
import importlib.util
import sys
from pathlib import Path
from .common import read,write,digest,Blocked


def activate(folder):
    entry=read(folder/'manifests/source_lock.json')['PriorTR']
    root=Path(entry['path'])
    for name,sha in entry['files'].items():
        if name.endswith('.py') and digest((root/name).read_bytes())!=sha:
            raise Blocked('Changed author source '+name)
    source=root/'image/Qwen3-VL'
    import transformers
    import transformers.models
    name='transformers.models.qwen3_vl'
    if any(n==name or n.startswith(name+'.') for n in sys.modules):
        raise Blocked('Qwen imported before frozen author activation')
    # Same package identity/relative imports as author's setup symlink, without
    # mutating the installed package or the source archive on disk.
    spec=importlib.util.spec_from_file_location(name,source/'qwen3/__init__.py',
                                             submodule_search_locations=[str(source/'qwen3')])
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    sys.path.insert(0,str(source))
    from transformers.models.qwen3_vl import modeling_qwen3_vl
    if Path(modeling_qwen3_vl.__file__).resolve()!=source/'qwen3/modeling_qwen3_vl.py':
        raise Blocked('Wrong Qwen implementation imported')
    return source


def run(folder,method,out,model_key):
    if out.exists() or out.with_suffix('.prefix.json').exists():
        raise Blocked('Use a new immutable smoke attempt')
    activate(folder)
    import torch
    from PIL import Image
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from visual_token_pruning import VTRConfig
    from visual_token_pruning.model.vtr_qwen3_vl import VTRQwen3VLForConditionalGeneration
    lock=read(folder/'manifests/model_weights_lock.json')[model_key]
    ratio={'Full':1.,'PriorTR50':.5,'PriorTR25':.25}[method]
    config=VTRConfig(enabled=method!='Full',strategy='priortr',prune_layer=3,
                     keep_ratio=ratio,keep_tokens=None,query_aggregation='question',head_aggregation='mean')
    model,info=VTRQwen3VLForConditionalGeneration.from_pretrained(
        lock['path'],dtype=torch.bfloat16,device_map={'':'cuda:0'},
        attn_implementation='eager',local_files_only=True,output_loading_info=True)
    write(out.with_suffix('.loading.json'),info)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise Blocked('Qwen checkpoint loading mismatch')
    # Follow the author's from_pretrained_vtr post-load setup exactly, while
    # retaining HF's loading_info tuple for strict tensor-loading acceptance.
    # Passing VTRConfig through HF kwargs leaks it into GenerationConfig.
    model.vtr_config=config
    model.vtr_strategy=model._create_strategy(config.strategy)
    from visual_token_pruning.strategy.clse import apply_clse_defaults
    apply_clse_defaults(config,model.config)
    if config.enabled:model._replace_text_model()
    model.eval().requires_grad_(False)
    processor=Qwen3VLProcessor.from_pretrained(lock['path'],local_files_only=True,
                                             min_pixels=262144,max_pixels=4194304)
    protocol=read(Path(__file__).resolve().parent.parent/'round3_spec.json')
    messages=[{'role':'user','content':[{'type':'image'},
        {'type':'text','text':'Describe this image.'+protocol['prompt']['suffix']}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    batch=processor(text=[text],images=[Image.new('RGB',(640,480),(127,100,80))],return_tensors='pt')
    batch={k:v.to(model.device) for k,v in batch.items()}
    full_visual=int((batch['input_ids']==model.config.image_token_id).sum())
    if batch['input_ids'].shape[1]>8192:raise Blocked('Prompt exceeds frozen limit')
    shapes={};handles=[]
    def hook(i):
        def record(module,args,kwargs):
            x=args[0] if args else kwargs['hidden_states']
            shapes.setdefault(str(i),[]).append(list(x.shape))
        return record
    for i,layer in enumerate(model.model.language_model.layers):
        handles.append(layer.register_forward_pre_hook(hook(i),with_kwargs=True))
    from .reference_stream import generate_native
    result=generate_native(model,batch,processor.tokenizer,
        {'job_key':'synthetic-'+model_key+'-'+method,'seed':1234,
         'generation':dict(max_new_tokens=8,temperature=1.,top_k=20,top_p=.95)},out.with_suffix('.prefix.json'))
    for h in handles:h.remove()
    write(out,dict(smoke_passed=True,native_gate_passed=False,production_ready=False,synthetic_only=True,
                   method=method,model_key=model_key,model_revision=lock['revision'],
                   full_visual_tokens=full_visual,layer_hidden_shapes=shapes,loading_info=info,
                   prompt_sha256=digest(text),result=result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',required=True);p.add_argument('--out',required=True)
    p.add_argument('--method',choices=['Full','PriorTR50','PriorTR25'],required=True)
    p.add_argument('--model-key',choices=['qwen8','qwen4'],default='qwen8')
    a=p.parse_args();run(Path(a.folder),a.method,Path(a.out),a.model_key)
