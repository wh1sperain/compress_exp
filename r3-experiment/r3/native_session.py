"""Persistent B1 author session. Certification is enforced by the runner.

No labels, altered stop rules, or research retries are introduced here.
"""
import functools
import importlib
import json
import io
import types
from pathlib import Path
from .common import read, digest, Blocked
from .control_adapters import uses_fastv_runtime,configure_fastv_control


def processor_hash_for_audit(config,expected):
    return ProcessorFingerprint(config).match(expected)


class ProcessorFingerprint:
    """Compute immutable image-processor fingerprints once per session."""
    def __init__(self,config):
        self.raw=digest(json.dumps(config,sort_keys=True))
        self.normalized=None
        # Frozen FastV's HF version serializes this API-validation list; HF4.37
        # does not. It does not participate in image transforms. Nothing else is
        # normalized, and the remaining whole dictionary must match the audit.
        known=['images','do_resize','size','resample','do_center_crop','crop_size',
           'do_rescale','rescale_factor','do_normalize','image_mean','image_std',
           'do_convert_rgb','return_tensors','data_format','input_data_format']
        canonical=dict(config)
        if canonical.get('_valid_processor_keys')==known:
            canonical.pop('_valid_processor_keys')
            self.normalized=digest(json.dumps(canonical,sort_keys=True))

    def match(self,expected):
        if self.raw==expected:return self.raw,self.raw
        if self.normalized is None:raise Blocked('Processor configuration changed')
        if self.normalized!=expected:raise Blocked('Processor transforms changed')
        return self.raw,self.normalized


class SessionInputFingerprints:
    """Tokenizer/processor objects are fixed after the native loader finishes."""
    def __init__(self,tokenizer,image_processor,qwen=False,chat_template=None):
        self.processor=ProcessorFingerprint(image_processor.to_dict())
        self.tokenizer_sha256=digest(tokenizer.backend_tokenizer.to_str()) if qwen else digest(json.dumps(tokenizer.get_vocab(),sort_keys=True))
        self.chat_template_sha256=digest(chat_template) if qwen else None


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
            if uses_fastv_runtime(self.condition):
                from .compat import llava_forward_with_cache_position
                self.model.forward=types.MethodType(llava_forward_with_cache_position,self.model)
        self.input_fingerprints=SessionInputFingerprints(self.tokenizer,
            self.processor.image_processor if model_key.startswith('qwen') else self.vision.image_processor,
            qwen=model_key.startswith('qwen'),chat_template=self.processor.chat_template if model_key.startswith('qwen') else None)

    def prepare(self,row):
        import torch
        from PIL import Image
        from .audit_qwen_inputs import question_text
        image_bytes=Path(row['image_path']).read_bytes()
        if digest(image_bytes)!=row['image_sha256']:
            raise Blocked('Input image changed')
        with Image.open(io.BytesIO(image_bytes)) as image:im=image.convert('RGB')
        text=question_text(row,self.spec['prompt']['suffix'])
        audit=row['native_inputs'].get(self.key)
        if not audit or not audit['eligible']:raise Blocked('Missing eligible native input audit')
        if self.key.startswith('qwen'):
            p=self.processor
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}]
            prompt=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            batch=p(text=[prompt],images=[im],return_tensors='pt')
            ids=batch['input_ids'][0]
            processor_raw,processor_sha=self.input_fingerprints.processor.match(audit['processor_sha256'])
            tokenizer_sha=self.input_fingerprints.tokenizer_sha256
            if self.input_fingerprints.chat_template_sha256!=audit['chat_template_sha256']:raise Blocked('Chat template changed')
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
            processor_raw,processor_sha=self.input_fingerprints.processor.match(audit['processor_sha256'])
            tokenizer_sha=self.input_fingerprints.tokenizer_sha256
            inputs={'inputs':ids.unsqueeze(0).to(self.model.device),'images':pixels}
            if uses_fastv_runtime(self.condition):
                control=configure_fastv_control(self.model,self.condition,
                    int((ids==constants.IMAGE_TOKEN_INDEX).nonzero()[0]),uid=row['uid'],
                    source_lock=self.folder/'manifests/source_lock.json')
        prompt_sha=digest(json.dumps(ids.tolist()))
        for key,value in [('prompt_sha256',prompt_sha),('processor_sha256',processor_sha),
                          ('tokenizer_sha256',tokenizer_sha),('visual_tokens',visual),('expanded_prompt_tokens',expanded)]:
            if key=='tokenizer_sha256' and self.key=='mini8' and key not in audit:
                continue  # Legacy Mini audit binds tokenizer via the frozen model source and prompt IDs.
            if audit.get(key)!=value:raise Blocked('Runtime input differs from audit: '+key)
        metadata={'prompt_sha256':prompt_sha,'prompt_ids':ids.tolist(),'full_visual_tokens':visual,'full_prompt_tokens':expanded,
                       'runtime_processor_sha256':processor_raw,'audited_processor_sha256':processor_sha}
        if uses_fastv_runtime(self.condition):metadata['control_evidence']=control
        return inputs,metadata

    def generate(self,row,job,checkpoint):
        from .reference_stream import generate_native
        from .correctness_probe import cache_lengths
        inputs,metadata=self.prepare(row)
        if self.condition=='Random25_L2' and job.get('config',{}).get('mask_seed')!=metadata['control_evidence']['mask_seed']:
            raise Blocked('Random control mask seed differs from registered job')
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
        return dict(result,**metadata,prefill_layer_kv=caches[0],final_layer_kv=caches[-1],
                    generation_eos_token_id=self.model.generation_config.eos_token_id)
