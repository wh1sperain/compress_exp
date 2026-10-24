"""Natural calibration with FastV's official composite HF LLaVA class.

The hybrid loader is used only to obtain strictly loaded source tensors. Every
tensor is mapped and checked into the independent official model; generation
never uses the hybrid wrapper. No teacher-forcing certification is rerun here.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from .common import Blocked,read,write,digest,code_hash
from .runner import canonical,publish_once
from .fastv_official_probe import mapped_name
from .natural_preparation import build


def remap_placeholder(ids,placeholder=-200):
    """Preserve ordinary text IDs; reject use of reserved composite image ID 0."""
    if ids.count(placeholder)!=1:raise Blocked('Expected exactly one image placeholder')
    if 0 in ids:raise Blocked('Image remapping collides with ordinary token ID 0')
    if any(t<0 and t!=placeholder for t in ids):raise Blocked('Unexpected negative token ID')
    return [0 if t==placeholder else t for t in ids],ids.index(placeholder)


def exhaustive_mapping(source_names,target_names):
    mapping={}
    for name in source_names:
        target=mapped_name(name)
        if target in mapping:raise Blocked('Duplicate target weight mapping')
        mapping[target]=name
    if set(mapping)!=set(target_names):raise Blocked('Non-exhaustive official tensor mapping')
    return mapping


class OfficialFastVSession:
    def __init__(self,folder,condition,receipt):
        import torch
        from .native_smoke import run as load_source
        if condition not in ('Full','FastV25'):raise Blocked('Unsupported reference condition')
        # Full uses the same frozen FastV HF runtime as the certified official path.
        source_model,self.tokenizer,vision,loading,_=load_source(folder,'FastVFull',receipt,load_only=True)
        from transformers.models.llava.configuration_llava import LlavaConfig
        from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
        from accelerate import init_empty_weights
        text=source_model.config.to_dict();text['model_type']='llama'
        config=LlavaConfig(text_config=text,vision_config=vision.vision_tower.config.to_dict(),
            image_token_index=0,pad_token_id=-1,vision_feature_layer=-2,vision_feature_select_strategy='default')
        config._attn_implementation='eager';config.text_config._attn_implementation='eager'
        with init_empty_weights():official=LlavaForConditionalGeneration(config)
        source=source_model.state_dict();mapping=exhaustive_mapping(source,official.state_dict())
        converted={k:source[v].detach().clone() for k,v in mapping.items()}
        official.load_state_dict(converted,strict=True,assign=True)
        official.to(device=source_model.device,dtype=torch.bfloat16).eval().requires_grad_(False)
        target=official.state_dict()
        if not all(torch.equal(target[k],source[v]) for k,v in mapping.items()):raise Blocked('Mapped weights changed')
        # Composite default GenerationConfig is not assumed to inherit nested text EOS.
        official.generation_config=copy.deepcopy(source_model.generation_config)
        self.model=official;self.condition=condition
        self.processor=vision.image_processor;self.image_config=source_model.config
        self.spec=read(Path(__file__).resolve().parent.parent/'round3_spec.json')
        publish_once(receipt.with_suffix('.mapping.json'),dict(count=len(mapping),all_exact=True,mapping=mapping,
            strict_source_loading=loading,generation_eos_token_id=official.generation_config.eos_token_id,
            scope='weight transfer only, no duplicate teacher diagnostic'))
        del source,target,converted,source_model,vision
        torch.cuda.empty_cache()

    def generate(self,row,job,checkpoint):
        import torch
        from PIL import Image
        from llava.mm_utils import process_images,tokenizer_image_token
        from llava.constants import DEFAULT_IMAGE_TOKEN,IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from .audit_qwen_inputs import question_text
        from .native_session import processor_hash_for_audit
        from .correctness_probe import cache_lengths
        from .reference_stream import generate_native
        if digest(Path(row['image_path']).read_bytes())!=row['image_sha256']:raise Blocked('Changed image')
        audit=row['native_inputs']['llava7']
        if not audit.get('eligible'):raise Blocked('Input audit ineligible')
        with Image.open(row['image_path']) as image:im=image.convert('RGB')
        conv=conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0],DEFAULT_IMAGE_TOKEN+'\n'+question_text(row,self.spec['prompt']['suffix']))
        conv.append_message(conv.roles[1],None)
        ids=tokenizer_image_token(conv.get_prompt(),self.tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').tolist()
        remapped,offset=remap_placeholder(ids,IMAGE_TOKEN_INDEX)
        raw_processor,processor_sha=processor_hash_for_audit(self.processor.to_dict(),audit['processor_sha256'])
        meta=dict(prompt_sha256=digest(json.dumps(ids)),processor_sha256=processor_sha,
            tokenizer_sha256=digest(json.dumps(self.tokenizer.get_vocab(),sort_keys=True)),
            visual_tokens=576,expanded_prompt_tokens=len(ids)-1+576)
        if meta['expanded_prompt_tokens']>2048 or any(audit.get(k)!=v for k,v in meta.items()):
            raise Blocked('Reference input differs from frozen audit')
        pixels=process_images([im],self.processor,self.image_config)
        if not isinstance(pixels,torch.Tensor):raise Blocked('Unexpected visual input tensor')
        tensor=torch.tensor([remapped],device=self.model.device,dtype=torch.long)
        inputs=dict(input_ids=tensor,attention_mask=torch.ones_like(tensor),
            pixel_values=pixels.to(device=self.model.device,dtype=torch.bfloat16))
        self.model.config.fastv_config=None if self.condition=='Full' else dict(fastv_k=2,fastv_r=.75,
            image_token_start_index=offset,image_token_length=576)
        caches=[]
        def hook(module,args,result):
            lengths=cache_lengths(result.past_key_values)
            if len(lengths)!=32:raise Blocked('Missing cache layers')
            if not caches:caches.append(lengths)
            elif len(caches)==1:caches.append(lengths)
            else:caches[-1]=lengths
        handle=self.model.register_forward_hook(hook)
        try:result=generate_native(self.model,inputs,self.tokenizer,job,checkpoint)
        finally:handle.remove()
        if not caches:raise Blocked('No forward cache evidence')
        return dict(result,**meta,prompt_ids=ids,hf_prompt_ids=remapped,
            generation_eos_token_id=self.model.generation_config.eos_token_id,
            image_token_start_index=offset,runtime_processor_sha256=raw_processor,
            prefill_layer_kv=caches[0],final_layer_kv=caches[-1],
            reference_entrypoint='frozen FastV LlavaForConditionalGeneration')


def plan(folder,condition):
    if condition not in ('Full','FastV25'):raise Blocked('Unsupported FastV reference condition')
    manifest=folder/'manifests/frozen-inputs-v1/calibration_manifest.jsonl'
    sha=digest(manifest.read_bytes())
    if read(manifest.parent/'manifest_lock.json').get('hashes',{}).get(manifest.name)!=sha:raise Blocked('Manifest changed')
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    spec_path=Path(__file__).resolve().parent.parent/'round3_spec.json'
    matrix=build(read(spec_path),rows,sha)
    jobs=[j for j in matrix['jobs'] if j['model']=='llava7' and j['condition']==condition and j['side']=='reference']
    identity=dict(manifest_sha256=sha,spec_sha256=digest(spec_path.read_bytes()),code_sha256=code_hash(),
        source_lock_sha256=digest((folder/'manifests/source_lock.json').read_bytes()),
        weights_lock_sha256=digest((folder/'manifests/model_weights_lock.json').read_bytes()))
    return jobs,{r['uid']:r for r in rows},identity


def run(folder,condition,out,integration_smoke=False):
    import fcntl
    jobs,rows,identity=plan(folder,condition);out.mkdir(parents=True,exist_ok=True)
    if integration_smoke:
        first={}
        for j in jobs:
            if j['repeat']==0:first.setdefault(j['task'],j['job_key'])
        jobs=[j for j in jobs if j['job_key'] in first.values()]
    with (folder/'correctness.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        session=None;done=0
        for job in jobs:
            directory=out/job['job_key'];directory.mkdir(exist_ok=True)
            binding=dict(job=job,input_sha256=canonical(rows[job['uid']]),identity=identity)
            binding_path=directory/'binding.json'
            if binding_path.exists():
                if read(binding_path)!=binding:raise Blocked('Original binding changed')
            else:publish_once(binding_path,binding)
            success=directory/'result.json'
            if success.exists():
                r=read(success)
                if r.get('binding_sha256')!=canonical(binding) or not r.get('record_complete'):raise Blocked('Invalid success')
                done+=1;continue
            if list(directory.glob('failure-*.json')):raise Blocked('Original failure requires explicit recovery')
            try:
                if session is None:session=OfficialFastVSession(folder,condition,out/('loading-'+str(time.time_ns())+'.json'))
                write(out/'status.json',dict(stage='running',completed=done,total=len(jobs),job_key=job['job_key'],updated_at=time.time()))
                generated=session.generate(rows[job['uid']],job,directory/'prefix.json')
                count=len(generated['generated_token_ids']);cap=job['generation']['max_new_tokens']
                if cap!=2048 or count!=generated['output_tokens'] or not 0<count<=cap:raise Blocked('Length/cap mismatch')
                if generated['output_cap_reached']!=(count==cap) or (not generated['eos_terminated'] and count!=cap):
                    raise Blocked('Termination mismatch')
                result=dict(job);result.update(generated,record_complete=True,binding_sha256=canonical(binding),
                    execution_identity=identity,completed_at=time.time(),native_gate_passed=False)
                publish_once(success,result);done+=1
            except BaseException as exc:
                publish_once(directory/('failure-'+str(time.time_ns())+'.json'),dict(error=str(exc)[:1200],
                    error_type=type(exc).__name__,job_key=job['job_key'],resampled=False))
                write(out/'status.json',dict(stage='failed',completed=done,total=len(jobs),updated_at=time.time()));raise
        receipt=dict(completed=done,total=len(jobs),exit_code=0,full_matrix_complete=not integration_smoke,native_gate_passed=False)
        path=out/('integration_smoke.json' if integration_smoke else 'completion.json')
        if path.exists():
            if read(path)!=receipt:raise Blocked('Completion changed')
        else:publish_once(path,receipt)
        write(out/'status.json',dict(receipt,stage='integration_smoke_complete' if integration_smoke else 'complete',updated_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--condition',choices=['Full','FastV25'],required=True);p.add_argument('--integration-smoke',action='store_true')
    a=p.parse_args();run(a.folder,a.condition,a.out,a.integration_smoke)
