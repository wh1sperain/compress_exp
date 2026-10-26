"""Thin, instrumented calls into fixed author sources. No selector reimplementation."""
import importlib.util
import math
import subprocess
import sys
import time
from pathlib import Path
from .core import read, sha, condition
from .interventions import choose

def verify_source(path, revision):
    path=Path(path).resolve()
    head=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=normal'],text=True)
    if head!=revision or dirty.strip():raise ValueError('Author source missing/changed: '+str(path))
    return path

def activate_prior(source):
    import transformers.models
    name='transformers.models.qwen3_vl'
    if any(k==name or k.startswith(name+'.') for k in sys.modules):
        raise ValueError('Use a fresh process per author environment')
    source=source/'image/Qwen3-VL'
    spec=importlib.util.spec_from_file_location(name,source/'qwen3/__init__.py',
        submodule_search_locations=[str(source/'qwen3')])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    sys.path.insert(0,str(source))

def vtr_parameters(method, ratio, spec, n=None):
    strategy={'Random':'fastv','Full':'priortr','Resolution':'priortr'}.get(method,method.lower())
    kw=dict(enabled=method not in ('Full','Resolution'),strategy=strategy,
            prune_layer=3,keep_ratio=ratio,query_aggregation='auto',head_aggregation='mean')
    if method=='SparseVLM':
        kw.update(prune_layer=spec['sparse_layers'],keep_ratio=[ratio**(1/3)]*3,
                  token_merge=False)
        if n is not None:kw['keep_tokens']=[max(1,round(n*ratio**(i/3))) for i in (1,2,3)]
    elif n is not None:kw['keep_tokens']=max(1,round(n*ratio))
    return kw

class Session:
    """One loaded model per condition; batch=1 reference, no silent optimized fallback."""
    def __init__(self, name, checkpoint, author_root, spec, locks, disabled=False, native=False, keep100=False):
        import torch
        self.name=name;self.method,self.ratio=condition(name);self.spec=spec;self.trace={}
        if keep100:self.ratio=1.
        if native and name!='Full':raise ValueError('Native reference is Full only')
        key='HiLo' if self.method=='HiLo' else 'PriorTR'
        source=verify_source(author_root,locks[key]['revision'])
        common=dict(local_files_only=True,device_map={'':'cuda:0'},
                    attn_implementation=spec['attention_backend'],output_loading_info=True)
        if native:
            from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
            model,info=Qwen3VLForConditionalGeneration.from_pretrained(checkpoint,dtype=torch.bfloat16,**common)
            processor=AutoProcessor.from_pretrained(checkpoint,local_files_only=True,**spec['processor'])
        elif key=='PriorTR':
            activate_prior(source)
            from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
            from visual_token_pruning import VTRConfig
            from visual_token_pruning.model.vtr_qwen3_vl import VTRQwen3VLForConditionalGeneration
            model,info=VTRQwen3VLForConditionalGeneration.from_pretrained(checkpoint,dtype=torch.bfloat16,**common)
            cfg=VTRConfig(**vtr_parameters(self.method,self.ratio,spec))
            if disabled:cfg.enabled=False
            model.vtr_config=cfg;model.vtr_strategy=model._create_strategy(cfg.strategy)
            if cfg.enabled:model._replace_text_model()
            processor=Qwen3VLProcessor.from_pretrained(checkpoint,local_files_only=True,**spec['processor'])
            self._observe_vtr(model)
        else:
            sys.path.insert(0,str(source))
            from transformers import AutoConfig,AutoProcessor
            from hilo_prune.models.qwen3_vl import Qwen3VLForConditionalGeneration
            import hilo_prune.models.qwen3_vl.modeling_hilo_qwen3_vl as hm
            cfg=AutoConfig.from_pretrained(checkpoint,local_files_only=True)
            cfg.text_config.hilo_keep_ratio=1. if disabled else self.ratio
            cfg.text_config.hilo_candidate_ratio=1. if disabled else min(1.,3*self.ratio)
            cfg.text_config.hilo_prune_layer=2;cfg.text_config.hilo_use_fusion=not disabled
            cfg.text_config.hilo_fusion_strength=.3;cfg.text_config.hilo_top_k_ratio=.1
            cfg.text_config.hilo_attention_threshold=.05;cfg.text_config.hilo_fusion_layers=[0,1]
            model,info=Qwen3VLForConditionalGeneration.from_pretrained(checkpoint,config=cfg,torch_dtype=torch.bfloat16,**common)
            processor=AutoProcessor.from_pretrained(checkpoint,local_files_only=True,**spec['processor'])
            original=hm.hierarchical_token_selection
            def observed(*args,**kw):
                t=time.perf_counter();value=original(*args,**kw)
                self.trace['selections'].append(dict(layer=2,original_indices=value.keep.cpu().tolist(),
                    fusion_indices=value.fusion.cpu().tolist(),candidate_indices=value.candidates.cpu().tolist(),
                    observer_wall_seconds=time.perf_counter()-t))
                return value
            hm.hierarchical_token_selection=observed
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Checkpoint loading mismatch')
        self.model=model.eval().requires_grad_(False);self.processor=processor
        self.handles=[]
        for i,layer in enumerate(model.model.language_model.layers):
            self.handles.append(layer.register_forward_pre_hook(self._layer_hook(i),with_kwargs=True))
        self.handles.append(model.register_forward_hook(self._cache_hook))

    def _observe_vtr(self,model):
        original=model.vtr_strategy.select_tokens
        def select(scores,num_tokens,config,layer_idx=0):
            import torch
            t=time.perf_counter()
            if self.method=='Random':
                k=config.get_keep_count(num_tokens,layer_idx)
                indices=torch.tensor(sorted(choose(range(num_tokens),k,[self.job['uid'],self.job['repeat'],'selector-v1'])),
                                     dtype=torch.long,device=scores.device)
            else:indices=original(scores,num_tokens,config,layer_idx)
            relative=indices.cpu().tolist()
            if self.override_indices is not None:
                if self.method not in ('FastV','PriorTR') or layer_idx!=0:raise ValueError('Rescue only single-stage')
                if len(self.override_indices)!=len(relative):raise ValueError('Rescue changed budget')
                if relative!=self.expected_original_indices:raise ValueError('Original selection drifted before rescue')
                relative=list(self.override_indices)
                indices=torch.tensor(relative,dtype=torch.long,device=scores.device)
            self.lineage=[self.lineage[i] for i in relative]
            self.trace['selections'].append(dict(stage=layer_idx,layer=config.get_prune_layers()[layer_idx],
                relative_indices=relative,original_indices=list(self.lineage),
                observer_wall_seconds=time.perf_counter()-t))
            return indices
        model.vtr_strategy.select_tokens=select

    def _layer_hook(self,i):
        def hook(module,args,kw):
            hidden=args[0] if args else kw['hidden_states']
            rec={'hidden_shape':list(hidden.shape)}
            for key in ('position_ids','cache_position','attention_mask'):
                x=kw.get(key)
                rec[key]=None if x is None else {'shape':list(x.shape)}
                if x is not None and key!='attention_mask' and self.forward_step in (0,31,1023,4095,32766):
                    rec[key]['values']=x.detach().cpu().tolist()
            self.trace['prefill_layers'].setdefault(str(i),rec)
            self.trace['last_layers'][str(i)]=rec
            if self.forward_step in (31,1023,4095,32766):
                self.trace.setdefault('fixed_step_layers',{}).setdefault(str(self.forward_step),{})[str(i)]=rec
        return hook

    def _cache_hook(self,module,args,result):
        cache=getattr(result,'past_key_values',None)
        if cache is not None:
            lengths=[int(x.keys.shape[-2]) for x in cache.layers] if hasattr(cache,'layers') else [int(x[0].shape[-2]) for x in cache]
            self.trace.setdefault('prefill_kv',lengths);self.trace['last_kv']=lengths
        self.forward_step+=1

    def prepare(self,row,job,override_indices=None,extra_fact=None,expected_original_indices=None):
        from PIL import Image
        from r3.audit_qwen_inputs import question_text
        if sha(Path(row['image_path']).read_bytes())!=row['image_sha256']:raise ValueError('Image SHA mismatch')
        self.job=job;self.override_indices=override_indices;self.forward_step=0
        self.expected_original_indices=expected_original_indices
        self.trace={'selections':[],'prefill_layers':{},'last_layers':{}}
        image=Image.open(row['image_path']).convert('RGB')
        text=question_text(row,self.spec['prompt']['suffix'])
        if extra_fact:text='Verified visual facts:\n'+extra_fact+'\n\n'+text
        prompt=self.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}],
            tokenize=False,add_generation_prompt=True)
        image_id=self.model.config.image_token_id
        def process(**kw):return self.processor(text=[prompt],images=[image],return_tensors='pt',**kw)
        batch=process();full_n=int((batch['input_ids']==image_id).sum())
        self.trace['full_visual_tokens']=full_n
        if self.method=='Resolution':
            # Search official processor pixel budget, measure actual merged tokens each time.
            target=max(1,round(full_n*self.ratio));lo,hi=1024,self.spec['processor']['max_pixels']
            best=None
            for _ in range(14):
                budget=(lo+hi)//2
                candidate=process(min_pixels=min(1024,budget),max_pixels=budget)
                n=int((candidate['input_ids']==image_id).sum())
                if best is None or (abs(n-target),n)<(best[0],best[1]):best=(abs(n-target),n,candidate)
                if n>target:hi=budget-1
                else:lo=budget+1
                if lo>hi:break
            batch=best[2];self.trace['resolution_target_tokens']=target
        actual=int((batch['input_ids']==image_id).sum());self.lineage=list(range(actual))
        if hasattr(self.model,'vtr_config'):
            cfg=self.model.vtr_config
            cfg.keep_tokens=vtr_parameters(self.method,self.ratio,self.spec,actual).get('keep_tokens')
        ids=batch['input_ids'][0].tolist()
        if len(ids)>self.spec['max_expanded_prompt_tokens']:raise ValueError('Input over frozen bound')
        self.trace.update(processed_visual_tokens=actual,image_grid_thw=batch['image_grid_thw'].tolist(),
            prompt_ids=ids,prompt_sha256=sha(ids),processor_sha256=sha(self.processor.image_processor.to_dict()),
            effective_visual_keep_ratio=actual/full_n if self.method=='Resolution' else None)
        self.trace['expected_final_visual_tokens']=actual if self.method in ('Full','Resolution') else max(1,round(full_n*self.ratio))
        extra={}
        if self.method=='HiLo':
            from hilo_prune.patches import image_to_merged_patch_features
            extra['image_tensor']=image_to_merged_patch_features(image,batch['image_grid_thw'][0].cpu(),
                patch_size=self.processor.image_processor.patch_size,merge_size=self.processor.image_processor.merge_size).to(self.model.device)
        return {k:v.to(self.model.device) for k,v in batch.items()},extra
