"""Persistent B1 author session. Certification is enforced by the runner.

No labels, altered stop rules, or research retries are introduced here.
"""
import functools
import importlib
import json
import types
from pathlib import Path
from .common import read, digest, Blocked


def processor_hash_for_audit(config,expected):
    raw=digest(json.dumps(config,sort_keys=True))
    if raw==expected:return raw,raw
    # Frozen FastV's HF version serializes this API-validation list; HF4.37
    # does not. It does not participate in image transforms. Nothing else is
    # normalized, and the remaining whole dictionary must match the audit.
    known=['images','do_resize','size','resample','do_center_crop','crop_size',
           'do_rescale','rescale_factor','do_normalize','image_mean','image_std',
           'do_convert_rgb','return_tensors','data_format','input_data_format']
    canonical=dict(config)
    if canonical.get('_valid_processor_keys')!=known:raise Blocked('Processor configuration changed')
    canonical.pop('_valid_processor_keys')
    normalized=digest(json.dumps(canonical,sort_keys=True))
    if normalized!=expected:raise Blocked('Processor transforms changed')
    return raw,normalized


class NativeSession:
    def __init__(self,folder,model_key,condition,receipt):
        self.folder=Path(folder);self.key=model_key
        self.condition=condition.removesuffix('_T0p6')
        self.spec=read(Path(__file__).resolve().parent.parent/'round3_spec.json')
        if model_key in ('qwen8','qwen4'):
            from .qwen_smoke import run
            self.model,self.processor,self.loading=run(self.folder,self.condition,Path(receipt),model_key,load_only=True)
            self.tokenizer=self.processor.tokenizer
        else:
            from .native_smoke import run
            self.model,self.tokenizer,self.vision,self.loading,self.paths=run(
                self.folder,self.condition,Path(receipt),load_only=True)
            if self.condition.startswith('FastV'):
                from .compat import llava_forward_with_cache_position
                self.model.forward=types.MethodType(llava_forward_with_cache_position,self.model)

    def prepare(self,row):
        import torch
        from PIL import Image
        from .audit_qwen_inputs import question_text
        if digest(Path(row['image_path']).read_bytes())!=row['image_sha256']:
            raise Blocked('Input image changed')
        with Image.open(row['image_path']) as image:im=image.convert('RGB')
        text=question_text(row,self.spec['prompt']['suffix'])
        audit=row['native_inputs'].get(self.key)
        if not audit or not audit['eligible']:raise Blocked('Missing eligible native input audit')
        if self.key.startswith('qwen'):
            p=self.processor
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}]
            prompt=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            batch=p(text=[prompt],images=[im],return_tensors='pt')
            ids=batch['input_ids'][0]
            processor_raw,processor_sha=processor_hash_for_audit(p.image_processor.to_dict(),audit['processor_sha256'])
            tokenizer_sha=digest(self.tokenizer.backend_tokenizer.to_str())
            if digest(p.chat_template)!=audit['chat_template_sha256']:raise Blocked('Chat template changed')
            visual=int((ids==self.model.config.image_token_id).sum())
            expanded=len(ids)
            inputs={k:v.to(self.model.device) for k,v in batch.items()}
        else:
            prefix='llavamini' if self.key=='mini8' else 'llava'
            mm=importlib.import_module(prefix+'.mm_utils')
            constants=importlib.import_module(prefix+'.constants')
            convs=importlib.import_module(prefix+'.conversation').conv_templates
            conv=convs['llava_llama_3_1' if self.key=='mini8' else 'llava_v1'].copy()
            conv.append_message(conv.roles[0],constants.DEFAULT_IMAGE_TOKEN+'\n'+text)
            conv.append_message(conv.roles[1],None)
            prompt=conv.get_prompt()
            ids=mm.tokenizer_image_token(prompt,self.tokenizer,constants.IMAGE_TOKEN_INDEX,return_tensors='pt')
            if int((ids==constants.IMAGE_TOKEN_INDEX).sum())!=1:raise Blocked('Image placeholder mismatch')
            pixels=mm.process_images([im],self.vision.image_processor,self.model.config)
            if not isinstance(pixels,torch.Tensor):raise Blocked('Unexpected image tensor path')
            pixels=pixels.to(device=self.model.device,dtype=torch.bfloat16)
            if self.key=='mini8':pixels=pixels.unsqueeze(0)
            visual=1 if self.key=='mini8' else 576
            expanded=len(ids)-1+visual
            processor_raw,processor_sha=processor_hash_for_audit(self.vision.image_processor.to_dict(),audit['processor_sha256'])
            tokenizer_sha=digest(json.dumps(self.tokenizer.get_vocab(),sort_keys=True))
            inputs={'inputs':ids.unsqueeze(0).to(self.model.device),'images':pixels}
            if self.condition.startswith('FastV'):
                # Position depends on each actual prompt; never reuse a previous question's index.
                keep={'FastV50':288,'FastV25':144}.get(self.condition)
                if keep is None:raise Blocked('Uncertified FastV condition')
                self.model.model.forward=functools.partial(self.model.model.fastv_forward,
                    fastv_config=dict(fastv_k=2,fastv_r=1-keep/576,
                        image_token_start_index=int((ids==constants.IMAGE_TOKEN_INDEX).nonzero()[0]),image_token_length=576))
        prompt_sha=digest(json.dumps(ids.tolist()))
        for key,value in [('prompt_sha256',prompt_sha),('processor_sha256',processor_sha),
                          ('tokenizer_sha256',tokenizer_sha),('visual_tokens',visual),('expanded_prompt_tokens',expanded)]:
            if key=='tokenizer_sha256' and self.key=='mini8' and key not in audit:
                continue  # Legacy Mini audit binds tokenizer via the frozen model source and prompt IDs.
            if audit.get(key)!=value:raise Blocked('Runtime input differs from audit: '+key)
        return inputs,{'prompt_sha256':prompt_sha,'full_visual_tokens':visual,'full_prompt_tokens':expanded,
                       'runtime_processor_sha256':processor_raw,'audited_processor_sha256':processor_sha}

    def generate(self,row,job,checkpoint):
        from .reference_stream import generate_native
        from .correctness_probe import cache_lengths
        inputs,metadata=self.prepare(row)
        caches=[]
        def capture(module,args,output):
            lengths=cache_lengths(output.past_key_values)
            if not caches:caches.append(lengths)
            elif len(caches)==1:caches.append(lengths)
            else:caches[-1]=lengths
        hook=self.model.register_forward_hook(capture)
        try:result=generate_native(self.model,inputs,self.tokenizer,job,checkpoint)
        finally:hook.remove()
        if not caches:raise Blocked('No cache evidence')
        return dict(result,**metadata,prefill_layer_kv=caches[0],final_layer_kv=caches[-1])
