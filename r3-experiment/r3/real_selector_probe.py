"""Observe real prefill, replay frozen author selectors on identical activations.

This is a selector-level check, not independent end-to-end model equivalence.
"""
import importlib
from pathlib import Path
from types import SimpleNamespace as NS
from .common import write,Blocked
from .native_sources import AuthorSources


def run(model,inputs,folder,method,out):
    import torch
    out=Path(out)
    if out.exists():raise Blocked('Keep diagnostic attempts immutable')
    out.mkdir(parents=True)
    sources=AuthorSources(folder/'manifests/source_lock.json');checks=[];handles=[];restore=[]
    tensors={};qwen=method.startswith('qwen-');name=method.removeprefix('qwen-')
    keep=144 if name.endswith('25') else 288
    def replace(obj,attr,fn):
        original=getattr(obj,attr);restore.append((obj,attr,original));setattr(obj,attr,fn);return original
    if name.startswith('PriorTR'):
        module=importlib.import_module('visual_token_pruning.strategy.priortr' if qwen else 'llava.vtr.strategy.priortr')
        cls=module.PriorTRStrategy;original=cls.compute_scores
        author=sources.priortr('qwen3' if qwen else 'llava')
        def score(self,attention,image_token_range,config,*args,**kw):
            actual=original(self,attention,image_token_range,config,*args,**kw)
            reference=author(self,attention,image_token_range,config,*args,**kw)
            start,end=image_token_range
            Q=attention[:,:,end,start:end].mean(1).squeeze(0).float()
            region=attention[:,:,end:-1,start:end] if qwen else attention[:,:,end:,start:end]
            P=region.mean(2).mean(1).squeeze(0).float();P=P/(P.sum()+1e-10);Q=Q/(Q.sum()+1e-10)
            formula=(P*torch.log((P+1e-10)/(Q+1e-10))).to(actual.dtype)
            k=int(config.keep_tokens) if config.keep_tokens else max(1,int((end-start)*config.keep_ratio))
            equal=torch.equal(actual,reference) and torch.equal(actual,formula)
            tensors.update(P=P.cpu(),Q=Q.cpu(),scores=actual.cpu(),topk=actual.topk(k).indices.cpu())
            checks.append({'component':'PriorTR scores','exact_author_replay':bool(equal),'image_range':[start,end],
                           'topk_count':k,'attention_shape':list(attention.shape),'query_suffix_included':True})
            if not equal:raise Blocked('Real PriorTR score differs from author/formula')
            return actual
        replace(cls,'compute_scores',score)
    elif name.startswith('Agile'):
        module=importlib.import_module('llava.model.llava_arch');author=sources.agile()
        old_rank=module.effective_rank;old_select=module.select_diverse_tokens_by_attention_and_distance
        def rank(features):
            actual=old_rank(features);ref=author['effective_rank'](features.clone())
            if not torch.equal(actual,ref):raise Blocked('Real effective rank differs')
            tensors['projected_features']=features.detach().cpu()
            checks.append({'component':'effective_rank','value':float(actual),'exact_author_replay':True})
            return actual
        def select(attention,distance,erank_input,**kw):
            actual=old_select(attention,distance,erank_input,**kw)
            reference=author['select_diverse_tokens_by_attention_and_distance'](attention,distance,erank_input,**kw)
            if not torch.equal(actual,reference):raise Blocked('Real Agile indices differ')
            tensors.update(cls_attention=attention.cpu(),cosine_distance=distance.cpu(),indices=actual.cpu())
            checks.append({'component':'Agile selection','exact_author_replay':True,'kept':int(actual.numel()),
                'unique':int(actual.unique().numel()),'tau_by_order':[author['calculate_adaptive_tau'](i,erank_input) for i in range(576)]})
            if actual.numel()!=keep:raise Blocked('Agile budget differs')
            return actual
        replace(module,'effective_rank',rank);replace(module,'select_diverse_tokens_by_attention_and_distance',select)
    elif name.startswith('VisionZip'):
        vision=model.get_vision_tower();captured={};author=sources.visionzip()
        def clip_hook(module,args,output):captured['clip']=output
        handles.append(vision.vision_tower.register_forward_hook(clip_hook))
        def vision_hook(module,args,output):
            class Replay:
                _info=vision.vision_tower._info
                vision_model=vision.vision_tower.vision_model
                def __call__(self,*a,**kw):return captured['clip']
            fake=NS(device=vision.device,dtype=vision.dtype,vision_tower=Replay())
            reference,indices=author(fake,args[0]);actual,actual_indices=output
            if not torch.equal(actual,reference) or not torch.equal(indices,actual_indices):raise Blocked('VisionZip replay differs')
            if actual.shape[1]!=keep or not bool((indices[:,0]==0).all()):raise Blocked('VisionZip CLS/budget differs')
            tensors.update(penultimate=captured['clip'].hidden_states[-2].cpu(),
                cls_attention=captured['clip'].attentions[-2][:,:,0,:].cpu(),
                metric=vision.vision_tower.vision_model.encoder.layers[-2].metric.cpu(),
                aggregated=actual.cpu(),dominant_indices=indices.cpu())
            captured['expected_projection']=model.model.mm_projector(reference)
            checks.append({'component':'VisionZip aggregation','exact_author_replay':True,'kept':keep,'cls_retained':True})
        handles.append(vision.register_forward_hook(vision_hook))
        def projector_hook(module,args,output):
            if 'expected_projection' not in captured:return
            if not torch.equal(output,captured['expected_projection']):raise Blocked('Projected merged features differ')
            checks.append({'component':'VisionZip projection','exact_author_replay':True,'shape':list(output.shape)})
        handles.append(model.model.mm_projector.register_forward_hook(projector_hook))
    else:raise Blocked('Unregistered real-selector diagnostic')
    try:
        with torch.inference_mode():model.generate(**inputs,max_new_tokens=1,do_sample=False,use_cache=True)
        if not checks:raise Blocked('Expected selector not called')
        write(out/'report.json',{'status':'PASS','native_gate_passed':False,'method':method,
            'scope':'same real activations replayed through frozen author functions; not independent end-to-end equivalence',
            'checks':checks,'source_revisions':{k:v['revision'] for k,v in sources.lock.items()}})
    except BaseException as exc:
        write(out/'report.json',{'status':'FAIL','native_gate_passed':False,'method':method,'checks':checks,'error':str(exc)[:1200]});raise
    finally:
        torch.save(tensors,out/'real_activations.pt')
        for h in handles:h.remove()
        for obj,attr,value in restore:setattr(obj,attr,value)
