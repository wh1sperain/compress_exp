"""Persistent A-track reference through frozen author builder entry points.

Independent loader/input integration, shared fixed author model forward and
protocol sampler. Never claims an independent compression reimplementation or
certification of a paged engine. One condition/native package per process.
"""
import argparse
import ast
import importlib
import json
import time
import types
from pathlib import Path
from .common import Blocked,read,write,digest,code_hash
from .mini_independent_probe import require_strict,finalize_bf16
from .natural_preparation import build
from .mini_natural_reference import run_jobs

CONDITIONS={'Full','PriorTR25','AgilePruner25','VisionZip25'}


def transform_builder(source,expected_count):
    tree=ast.parse(source)
    class BF16(ast.NodeTransformer):
        count=0
        def visit_Attribute(self,node):
            self.generic_visit(node)
            if isinstance(node.value,ast.Name) and node.value.id=='torch' and node.attr=='float16':
                node.attr='bfloat16';self.count+=1
            return node
    change=BF16();tree=change.visit(tree)
    if change.count!=expected_count:raise Blocked('Frozen builder dtype sites changed')
    return ast.fix_missing_locations(tree),change.count


def guard_resize(model):
    original=model.resize_token_embeddings
    def checked(self,new_num_tokens=None,*args,**kwargs):
        count=int(self.get_input_embeddings().weight.shape[0])
        if new_num_tokens is not None and new_num_tokens!=count:
            raise Blocked('Author builder attempted to resize frozen embeddings')
        return original(new_num_tokens,*args,**kwargs)
    model.resize_token_embeddings=types.MethodType(checked,model)
    return original


def load_author(folder,condition,out):
    if condition not in CONDITIONS:raise Blocked('Unsupported A reference condition')
    from .native_smoke import activate
    from .native_sources import visionzip_arguments
    paths=activate(Path(folder),condition)
    import torch
    prior=condition=='PriorTR25'
    module_name='llava.vtr.model.builder' if prior else 'llava.model.builder'
    builder=importlib.import_module(module_name)
    language=importlib.import_module('llava.model.language_model.llava_llama')
    cls=importlib.import_module('llava.vtr.model.priortr_llava').PriorTRLlava if prior else language.LlavaLlamaForCausalLM
    lock=read(Path(folder)/'manifests/model_weights_lock.json');model_path=lock['llava7']['path']
    cfg=language.LlavaConfig.from_pretrained(model_path,local_files_only=True)
    cfg.mm_vision_tower=lock['clip336']['path'];cfg.image_aspect_ratio='pad'
    author_root=paths['AgilePruner'] if condition=='AgilePruner25' else paths['PriorTR']/'image/LLaVA'
    source_path=author_root/('llava/vtr/model/builder.py' if prior else 'llava/model/builder.py')
    source=source_path.read_text(encoding='utf-8');tree,count=transform_builder(source,4 if prior else 5)
    ns=dict(vars(builder));info_saved={};original_resize=[]
    class StrictLoader:
        @staticmethod
        def from_pretrained(path,**kwargs):
            if str(path)!=str(model_path):raise Blocked('Unexpected author checkpoint base')
            kwargs.update(config=cfg,local_files_only=True,output_loading_info=True,attn_implementation='eager')
            model,info=cls.from_pretrained(path,**kwargs)
            write(out/'loading_info.json',info);require_strict(info);info_saved.update(info)
            original_resize.append((model,guard_resize(model)))
            return model
    exec(compile(tree,str(source_path),'exec'),ns)
    if prior:ns['VTR_MODEL_CLASSES']=dict(ns['VTR_MODEL_CLASSES'],priortr=StrictLoader)
    else:ns['LlavaLlamaForCausalLM']=StrictLoader
    kwargs=dict(device_map='cuda:0',device='cuda',local_files_only=True,attn_implementation='eager')
    if condition=='AgilePruner25':kwargs['visual_token_num']=144
    try:
        if prior:tokenizer,model,processor,context=ns['load_vtr_model'](model_path,model_base=None,model_name='llava-v1.5-7b',model_type='priortr',**kwargs)
        else:tokenizer,model,processor,context=ns['load_pretrained_model'](model_path,None,'llava-v1.5-7b',**kwargs)
    finally:
        for loaded,original in original_resize:loaded.resize_token_embeddings=original
    model.eval().requires_grad_(False);conversion=finalize_bf16(model,'cuda:0')
    if len(tokenizer)!=model.get_input_embeddings().weight.shape[0]:raise Blocked('Tokenizer size changed')
    if prior:
        config=importlib.import_module('llava.vtr.config').VTRConfig
        model.setup_vtr(config(enabled=True,strategy='priortr',prune_layer=3,keep_tokens=144,query_aggregation='question',head_aggregation='mean'))
    if condition=='VisionZip25':
        from visionzip import visionzip
        model=visionzip(model,**visionzip_arguments(144))
    receipt=dict(condition=condition,model_revision=lock['llava7']['revision'],clip_revision=lock['clip336']['revision'],
        builder_sha256=digest(source),dtype_ast_sites=count,loading_info=info_saved,dtype_conversion=conversion,
        author_context_return=context,supports_cache_class=getattr(model,'_supports_cache_class',None),
        adjustments=['study BF16 conversion','local pinned CLIP path','pad image aspect ratio','strict checkpoint load and no embedding resize'],
        native_gate_passed=False,scope='author builder integration; shared fixed author forward and protocol sampler')
    write(out/'loading_receipt.json',receipt)
    return model,tokenizer,processor,receipt


class LlavaAuthorSession:
    def __init__(self,folder,condition,loading_dir):
        self.condition=condition;loading_dir=Path(loading_dir);loading_dir.mkdir(parents=True,exist_ok=False)
        self.model,self.tokenizer,self.processor,self.loading=load_author(folder,condition,loading_dir)
        self.spec=read(Path(__file__).resolve().parent.parent/'round3_spec.json')

    def prepare(self,row):
        import torch
        from PIL import Image
        from llava.constants import DEFAULT_IMAGE_TOKEN,IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images,tokenizer_image_token
        from .audit_qwen_inputs import question_text
        if digest(Path(row['image_path']).read_bytes())!=row['image_sha256']:raise Blocked('Image changed')
        audit=row['native_inputs']['llava7']
        conv=conv_templates['llava_v1'].copy();conv.append_message(conv.roles[0],DEFAULT_IMAGE_TOKEN+'\n'+question_text(row,self.spec['prompt']['suffix']));conv.append_message(conv.roles[1],None)
        ids=tokenizer_image_token(conv.get_prompt(),self.tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt')
        with Image.open(row['image_path']) as image:pixels=process_images([image.convert('RGB')],self.processor,self.model.config)
        if not isinstance(pixels,torch.Tensor) or int((ids==IMAGE_TOKEN_INDEX).sum())!=1:raise Blocked('Unsupported image input')
        values=dict(prompt_sha256=digest(json.dumps(ids.tolist())),processor_sha256=digest(json.dumps(self.processor.to_dict(),sort_keys=True)),
            tokenizer_sha256=digest(json.dumps(self.tokenizer.get_vocab(),sort_keys=True)),visual_tokens=576,
            expanded_prompt_tokens=len(ids)-1+576,processed_size=list(pixels.shape[-2:]))
        for key,value in values.items():
            if audit.get(key)!=value:raise Blocked('Author input differs from audit: '+key)
        return {'inputs':ids.unsqueeze(0).to('cuda:0'),'images':pixels.to(device='cuda:0',dtype=torch.bfloat16)},values

    def generate(self,row,job,checkpoint):
        from .reference_stream import generate_native
        from .correctness_probe import cache_lengths
        inputs,metadata=self.prepare(row);kv=[]
        def capture(module,args,output):
            lengths=cache_lengths(output.past_key_values)
            if not kv:kv.append(lengths)
            elif len(kv)==1:kv.append(lengths)
            else:kv[-1]=lengths
        hook=self.model.register_forward_hook(capture)
        try:result=generate_native(self.model,inputs,self.tokenizer,job,checkpoint)
        finally:hook.remove()
        expected=576 if self.condition=='Full' else 144
        actual=kv[0][-1]-(len(inputs['inputs'][0])-1) if kv else None
        if actual!=expected or len(kv[0])!=32:raise Blocked('Actual A visual/KV budget differs')
        return dict(result,prompt_ids=inputs['inputs'][0].cpu().tolist(),prompt_sha256=metadata['prompt_sha256'],
            generation_eos_token_id=self.model.generation_config.eos_token_id,full_visual_tokens=576,actual_visual_tokens=actual,
            prefill_layer_kv=kv[0],final_layer_kv=kv[-1],reference_loading=self.loading)


def prepare_jobs(spec,rows,identity,condition,smoke=False,integration_smoke=False):
    if smoke and integration_smoke:raise Blocked('Choose single-image or five-task smoke, not both')
    if condition not in CONDITIONS:raise Blocked('Unsupported A reference condition')
    plan=build(spec,rows,identity['manifest_sha256'])
    jobs=[j for j in plan['jobs'] if j['model']=='llava7' and j['condition']==condition and j['side']=='reference']
    if integration_smoke:
        first={}
        for job in jobs:
            if job['repeat']==0:first.setdefault(job['task'],job['job_key'])
        return [job for job in jobs if job['job_key'] in first.values()]
    return jobs[:1] if smoke else jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--condition',choices=sorted(CONDITIONS),required=True);p.add_argument('--out',type=Path,required=True)
    smoke_flags=p.add_mutually_exclusive_group()
    smoke_flags.add_argument('--smoke',action='store_true');smoke_flags.add_argument('--integration-smoke',action='store_true')
    p.add_argument('--numeric-receipt',type=Path);a=p.parse_args()
    if a.manifest.name!='calibration_manifest.jsonl':raise Blocked('Frozen calibration only')
    if not (a.smoke or a.integration_smoke):
        if a.numeric_receipt is None:raise Blocked('Full calibration requires independent condition numeric receipt')
        proof=read(a.numeric_receipt)
        if proof.get('status')!='PASS' or proof.get('condition')!=a.condition:raise Blocked('Numeric receipt missing or wrong condition')
    spec_path=Path(__file__).resolve().parent.parent/'round3_spec.json';rows=[json.loads(x) for x in a.manifest.read_text().splitlines()]
    identity=dict(manifest_sha256=digest(a.manifest.read_bytes()),spec_sha256=digest(spec_path.read_bytes()),code_sha256=code_hash(),
        source_lock_sha256=digest((a.folder/'manifests/source_lock.json').read_bytes()),weights_lock_sha256=digest((a.folder/'manifests/model_weights_lock.json').read_bytes()),
        reference_implementation='r3.llava_natural_reference.LlavaAuthorSession',backend='author-native-hf-b1')
    if a.numeric_receipt:identity['numeric_receipt_sha256']=digest(a.numeric_receipt.read_bytes())
    jobs=prepare_jobs(read(spec_path),rows,identity,a.condition,a.smoke,a.integration_smoke)
    import fcntl
    with (a.folder/'correctness.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=run_jobs(jobs,rows,a.out,lambda:LlavaAuthorSession(a.folder,a.condition,a.out/('loading-'+str(time.time_ns()))),identity=identity)
        if a.integration_smoke:
            write(a.out/'integration_smoke.json',dict(completed=done,total=5,exit_code=0,full_matrix_complete=False,
                stage='integration_smoke_complete',native_gate_passed=False,scope='five tasks first calibration job, seed0, original cap'))

if __name__=='__main__':main()
