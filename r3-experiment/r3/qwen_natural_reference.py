"""Label-free natural calibration via the pinned author's public Qwen loader.

Separate from NativeSession construction. Same request-local inverse-CDF sampler
and unmodified natural cap/EOS. A smoke selects one registered image/repeat but
does not shorten its generation. This worker never grants a native/backend gate.
"""
import argparse
import json
import os
import time
from pathlib import Path
from .common import read,write,digest,seed,code_hash,Blocked
from .runner import publish_once,canonical
from .natural_preparation import validate_calibration,build
from .manifest import TASKS


def jobs_for(spec,rows,manifest_sha,model,condition,uid=None,repeat=None):
    validate_calibration(rows)
    if model not in ('qwen8','qwen4') or condition not in ('Full','PriorTR25'):
        raise Blocked('Unsupported natural reference condition')
    expected=dict(temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,min_new_tokens=0)
    if any(spec['generation'].get(k)!=v for k,v in expected.items()):raise Blocked('Changed generation protocol')
    if spec['models'][model]['max_new_tokens']!=32768:raise Blocked('Changed Qwen output cap')
    if (uid is None)!=(repeat is None):raise Blocked('Smoke requires both UID and repeat')
    if uid is not None and (uid not in {r['uid'] for r in rows} or repeat not in range(3)):
        raise Blocked('Unregistered smoke selection')
    matrix=build(spec,rows,manifest_sha,include_qwen4=model=='qwen4')
    return [j for j in matrix['jobs'] if j['model']==model and j['condition']==condition and j['side']=='reference'
            and (uid is None or (j['uid']==uid and j['repeat']==repeat))]


class AuthorQwenSession:
    def __init__(self,folder,model_key,condition):
        import torch
        from .qwen_smoke import activate
        activate(folder)
        from visual_token_pruning import VTRConfig
        from visual_token_pruning.model.vtr_qwen3_vl import VTRQwen3VLForConditionalGeneration
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
        lock=read(folder/'manifests/model_weights_lock.json')[model_key]
        config=VTRConfig(enabled=condition!='Full',strategy='priortr',prune_layer=3,
            keep_ratio=1. if condition=='Full' else .25,keep_tokens=None,
            query_aggregation='question',head_aggregation='mean')
        self.model=VTRQwen3VLForConditionalGeneration.from_pretrained_vtr(lock['path'],vtr_config=config,
            dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',local_files_only=True)
        self.model.eval().requires_grad_(False)
        self.processor=Qwen3VLProcessor.from_pretrained(lock['path'],local_files_only=True,
            min_pixels=262144,max_pixels=4194304)
        self.key=model_key
        self.spec=read(Path(__file__).resolve().parent.parent/'round3_spec.json')

    def generate(self,row,job,checkpoint):
        from PIL import Image
        from .audit_qwen_inputs import question_text
        from .reference_stream import generate_native
        from .correctness_probe import cache_lengths
        if digest(Path(row['image_path']).read_bytes())!=row['image_sha256']:raise Blocked('Changed image')
        audit=row['native_inputs'].get(self.key)
        if not audit or not audit.get('eligible'):raise Blocked('Missing native input audit')
        with Image.open(row['image_path']) as source:image=source.convert('RGB')
        p=self.processor
        text=p.apply_chat_template([{'role':'user','content':[{'type':'image'},
            {'type':'text','text':question_text(row,self.spec['prompt']['suffix'])}]}],tokenize=False,add_generation_prompt=True)
        batch=p(text=[text],images=[image],return_tensors='pt')
        ids=batch['input_ids'][0].tolist()
        metadata=dict(prompt_sha256=digest(json.dumps(ids)),
            processor_sha256=digest(json.dumps(p.image_processor.to_dict(),sort_keys=True)),
            tokenizer_sha256=digest(p.tokenizer.backend_tokenizer.to_str()),chat_template_sha256=digest(p.chat_template),
            expanded_prompt_tokens=len(ids),visual_tokens=ids.count(self.model.config.image_token_id))
        if len(ids)>8192 or any(audit.get(k)!=v for k,v in metadata.items()):raise Blocked('Input differs from frozen audit')
        batch={k:v.to(self.model.device) for k,v in batch.items()}
        caches=[]
        def capture(module,args,result):
            lengths=cache_lengths(result.past_key_values)
            if len(lengths)!=36:raise Blocked('Incomplete cache layers')
            if not caches:caches.append(lengths)
            elif len(caches)==1:caches.append(lengths)
            else:caches[-1]=lengths
        handle=self.model.register_forward_hook(capture)
        try:result=generate_native(self.model,batch,p.tokenizer,job,checkpoint)
        finally:handle.remove()
        if not caches:raise Blocked('Missing generation evidence')
        eos=self.model.generation_config.eos_token_id
        return dict(result,**metadata,prompt_ids=ids,generation_eos_token_id=eos,
            prefill_layer_kv=caches[0],final_layer_kv=caches[-1],
            reference_entrypoint='VTRQwen3VLForConditionalGeneration.from_pretrained_vtr')


def run_jobs(jobs,rows,identity,out,session_factory):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    session=None;completed=0
    for job in jobs:
        directory=out/job['job_key'];directory.mkdir(exist_ok=True)
        row=rows[job['uid']]
        binding=dict(job=job,input_sha256=canonical(row),identity=identity)
        path=directory/'binding.json'
        if path.exists():
            if read(path)!=binding:raise Blocked('Changed natural job binding')
        else:publish_once(path,binding)
        success=directory/'result.json'
        if success.exists():
            result=read(success)
            if not result.get('record_complete') or result.get('binding_sha256')!=canonical(binding):
                raise Blocked('Invalid existing success')
            completed+=1;continue
        if list(directory.glob('failure-*.json')):raise Blocked('Preserved failure needs explicit validated recovery')
        try:
            if session is None:session=session_factory()
            write(out/'status.json',dict(stage='running',job_key=job['job_key'],completed=completed,total=len(jobs),updated_at=time.time()))
            generated=session.generate(row,job,directory/'prefix.json')
            count=len(generated['generated_token_ids'])
            if count!=generated['output_tokens'] or not 0<count<=32768:raise Blocked('Invalid generation length')
            if generated['output_cap_reached']!=(count==32768):raise Blocked('Wrong cap flag')
            if not generated['eos_terminated'] and count!=32768:raise Blocked('Unexpected early stop')
            result=dict(job,**generated)
            result.update(record_complete=True,binding_sha256=canonical(binding),completed_at=time.time(),
                scoring_status='pending_separate_label_process',native_gate_passed=False,execution_identity=identity)
            publish_once(success,result);completed+=1
        except BaseException as exc:
            publish_once(directory/('failure-'+str(time.time_ns())+'.json'),dict(job_key=job['job_key'],
                error=str(exc)[:1200],type=type(exc).__name__,resampled=False))
            write(out/'status.json',dict(stage='failed',completed=completed,total=len(jobs),updated_at=time.time()))
            raise
    receipt=dict(completed=completed,total=len(jobs),exit_code=0,native_gate_passed=False,
                 full_matrix_complete=len(jobs)==300)
    path=out/('completion.json' if len(jobs)==300 else 'integration_smoke.json')
    if path.exists():
        if read(path)!=receipt:raise Blocked('Previous completion differs')
    else:publish_once(path,receipt)
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['qwen8','qwen4'],required=True)
    p.add_argument('--condition',choices=['Full','PriorTR25'],required=True)
    p.add_argument('--uid');p.add_argument('--repeat',type=int)
    p.add_argument('--integration-smoke',action='store_true')
    p.add_argument('--supplemental-audit');p.add_argument('--supplemental-status')
    a=p.parse_args();folder=a.folder
    manifest=folder/'manifests/frozen-inputs-v1/calibration_manifest.jsonl'
    lock=read(manifest.parent/'manifest_lock.json')
    if digest(manifest.read_bytes())!=lock.get('hashes',{}).get(manifest.name):raise Blocked('Calibration lock mismatch')
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    extra={}
    if a.model=='qwen4':
        from .supplemental_inputs import attach
        if not a.supplemental_audit and not a.supplemental_status:
            supplemental=read(folder/'telemetry/qwen4_supplemental_audit_launch.json')
            a.supplemental_audit,a.supplemental_status=supplemental['out'],supplemental['status_path']
        if not a.supplemental_audit or not a.supplemental_status:raise Blocked('Qwen4 requires supplemental audit')
        rows,extra=attach(rows,a.supplemental_audit,a.supplemental_status)
    spec_path=Path(__file__).resolve().parent.parent/'round3_spec.json'
    spec=read(spec_path)
    jobs=jobs_for(spec,rows,digest(manifest.read_bytes()),a.model,a.condition,a.uid,a.repeat)
    if a.integration_smoke:
        if a.uid is not None:raise Blocked('Choose single-row or five-task smoke')
        first={}
        for job in jobs:
            if job['repeat']==0:first.setdefault(job['task'],job['job_key'])
        jobs=[j for j in jobs if j['job_key'] in first.values()]
    identity=dict(code_sha256=code_hash(),spec_sha256=digest(spec_path.read_bytes()),manifest_sha256=digest(manifest.read_bytes()),
        source_lock_sha256=digest((folder/'manifests/source_lock.json').read_bytes()),
        weights_lock_sha256=digest((folder/'manifests/model_weights_lock.json').read_bytes()),
        scope='natural calibration reference only',**extra)
    import fcntl
    with (folder/'correctness.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_jobs(jobs,{r['uid']:r for r in rows},identity,a.out,
            lambda:AuthorQwenSession(folder,a.model,a.condition))


if __name__=='__main__':main()
