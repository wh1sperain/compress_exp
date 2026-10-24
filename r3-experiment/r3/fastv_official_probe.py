"""Independent official HF LLaVA entry vs hybrid wrapper, fixed author source.

Maps every checkpoint tensor losslessly into a separate model instance. The
image placeholder is remapped to unused vocabulary ID 0 (overwritten by image
features); ordinary text tokens and vocabulary size are unchanged.
"""
import argparse,json,time
from pathlib import Path
from .common import read,write,Blocked,digest,code_hash


def mapped_name(name):
    if name.startswith('model.vision_tower.vision_tower.'):
        return 'vision_tower.'+name[len('model.vision_tower.vision_tower.'):]
    for source,target in [('model.mm_projector.0.','multi_modal_projector.linear_1.'),
                          ('model.mm_projector.2.','multi_modal_projector.linear_2.')]:
        if name.startswith(source):return target+name[len(source):]
    if name.startswith(('model.embed_tokens.','model.layers.','model.norm.','lm_head.')):
        return 'language_model.'+name
    raise Blocked('Unmapped source tensor: '+name)


def run(folder,out):
    import torch
    from .native_session import NativeSession
    from .correctness_probe import cache_lengths
    from .model_resume_probe import diagnostic_no_eos
    out=Path(out)
    if out.exists():raise Blocked('Existing official comparison attempt')
    out.mkdir(parents=True)
    folder=Path(folder)
    manifest=folder/'manifests/frozen-inputs-v1/calibration_manifest.jsonl'
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    row=min((r for r in rows if r['task']=='DocVQA'),key=lambda r:digest(r['uid']))
    session=NativeSession(folder,'llava7','FastV50',out/'loading.json')
    hybrid=session.model
    inputs,meta=session.prepare(row)
    # Import the author's official composite class, not PriorTR's wrapper.
    from transformers.models.llava.configuration_llava import LlavaConfig
    from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
    from transformers import LogitsProcessorList
    from accelerate import init_empty_weights
    text=hybrid.config.to_dict();text['model_type']='llama'
    vision=hybrid.get_vision_tower().vision_tower.config.to_dict()
    config=LlavaConfig(text_config=text,vision_config=vision,image_token_index=0,
                       pad_token_id=-1,vision_feature_layer=-2,vision_feature_select_strategy='default')
    config._attn_implementation='eager';config.text_config._attn_implementation='eager'
    with init_empty_weights():official=LlavaForConditionalGeneration(config)
    source=hybrid.state_dict();target=official.state_dict();mapping={mapped_name(k):k for k in source}
    if set(mapping)!=set(target):
        write(out/'mapping_failure.json',{'missing':sorted(set(target)-set(mapping)),
                                         'unexpected':sorted(set(mapping)-set(target))})
        raise Blocked('Official checkpoint tensor mapping is not exhaustive')
    converted={k:source[v].detach().clone() for k,v in mapping.items()}
    official.load_state_dict(converted,strict=True,assign=True)
    official.to(device=hybrid.device,dtype=torch.bfloat16).eval().requires_grad_(False)
    if not all(torch.equal(official.state_dict()[k],source[v]) for k,v in mapping.items()):
        raise Blocked('Mapped tensor values changed')
    write(out/'tensor_mapping.json',{'count':len(mapping),'all_exact':True,'mapping':mapping})
    ids=inputs['inputs'].clone()
    if bool((ids==0).any()):raise Blocked('Placeholder remap collides with existing text token')
    ids[ids==-200]=0
    official_inputs={'input_ids':ids,'pixel_values':inputs['images'],'attention_mask':torch.ones_like(ids)}
    forcing=session.tokenizer.encode(' The image contains information that can be examined carefully.',add_special_tokens=False)
    forcing=(forcing*(1025//len(forcing)+1))[:1025]
    all_reports=[]
    original_forward=hybrid.model.forward
    import functools
    try:
        for method,keep in [('Full',576),('FastV50',288),('FastV25',144)]:
            cfg=dict(fastv_k=2,fastv_r=1-keep/576,image_token_start_index=int((ids==0).nonzero()[0,1]),image_token_length=576)
            if method=='Full':
                # Native class forward, before the method-specific partial.
                hybrid.model.forward=type(hybrid.model).forward.__get__(hybrid.model,type(hybrid.model))
                official.config.fastv_config=None
            else:
                hybrid.model.forward=functools.partial(hybrid.model.fastv_forward,fastv_config=cfg)
                official.config.fastv_config=cfg
            snapshots={};kv={}
            for name,model,batch in [('hybrid',hybrid,inputs),('official',official,official_inputs)]:
                write(out/'status.json',{'stage':'teacher_forcing','method':method,'path':name,'updated_at':time.time()})
                logits=[];lengths=[];step=[0]
                def capture(module,args,result):lengths.append(cache_lengths(result.past_key_values))
                hook=model.register_forward_hook(capture)
                class Force:
                    def __call__(self,unused,scores):
                        logits.append(scores[0].detach().float().cpu())
                        forced=torch.full_like(scores,-float('inf'));forced[:,forcing[step[0]]]=0
                        step[0]+=1;return forced
                try:
                    with diagnostic_no_eos(model) as gc,torch.inference_mode():
                        model.generate(**batch,generation_config=gc,max_new_tokens=1025,do_sample=False,
                            use_cache=True,logits_processor=LogitsProcessorList([Force()]))
                finally:hook.remove()
                if step[0]!=1025:raise Blocked('Official diagnostic ended early')
                snapshots[name]=logits;kv[name]=lengths
            comparisons=[]
            for i,(a,b) in enumerate(zip(snapshots['hybrid'],snapshots['official'])):
                p=a.softmax(-1);q=b.softmax(-1)
                comparisons.append({'step':i,'kl':float((p*(p.clamp_min(1e-30).log()-q.clamp_min(1e-30).log())).sum()),
                    'tv':float((p-q).abs().sum()/2),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                    'argmax_equal':bool(a.argmax()==b.argmax()),'kv_equal':kv['hybrid'][i]==kv['official'][i]})
            passed=all(c['finite'] and c['kl']<.02 and c['tv']<.08 and c['kv_equal'] for c in comparisons)
            report={'method':method,'status':'PASS' if passed else 'FAIL','steps':1025,
                    'max_kl':max(c['kl'] for c in comparisons),'max_tv':max(c['tv'] for c in comparisons),
                    'argmax_mismatches':sum(not c['argmax_equal'] for c in comparisons),
                    'kv_mismatches':sum(not c['kv_equal'] for c in comparisons)}
            write(out/(method+'.json'),{'summary':report,'comparisons':comparisons,'layer_kv':kv})
            all_reports.append(report)
        write(out/'report.json',{'status':'PASS' if all(r['status']=='PASS' for r in all_reports) else 'FAIL',
            'native_gate_passed':False,'scope':'single calibration image, official full entry, prefill plus 1024 forced decode steps; natural/backend gates remain',
            'code_sha256':code_hash(),'uid':row['uid'],'image_sha256':row['image_sha256'],**meta,'conditions':all_reports})
        write(out/'status.json',{'stage':'complete','conditions':all_reports,'updated_at':time.time()})
    finally:hybrid.model.forward=original_forward


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.folder,a.out)
