"""Mini natural reference calibration via reviewed author builder, not NativeSession.

Shares the required request-local sampler and fixed author model class. This is
an independent loader/input integration path, NOT a new decoding engine. A
comparison to NativeSession therefore cannot certify an unimplemented paged
backend. Generates calibration only; no labels are read and no native gate issued.
"""
import argparse
import json
import os
import time
from pathlib import Path
from .common import Blocked,read,write,digest,seed,code_hash
from .natural_preparation import validate_calibration,build as build_natural
from .runner import publish_once,canonical


class MiniAuthorSession:
    def __init__(self,folder,loading_dir):
        from .mini_independent_probe import author_load,verify_vision_checkpoint
        folder=Path(folder);loading_dir=Path(loading_dir);loading_dir.mkdir(parents=True,exist_ok=False)
        self.model,self.tokenizer,self.processor,self.loading=author_load(folder,loading_dir)
        lock=read(folder/'manifests/model_weights_lock.json')
        self.loading['vision_checkpoint_proof']=verify_vision_checkpoint(self.model,lock['mini8']['path'])
        self.loading['model_revision']=lock['mini8']['revision']
        write(loading_dir/'strict_loading.json',self.loading)

    def generate(self,row,job,checkpoint):
        from .mini_independent_probe import author_prepare
        from .reference_stream import generate_native
        from .correctness_probe import cache_lengths
        inputs=author_prepare(self.model,self.tokenizer,self.processor,row)
        audit=row['native_inputs']['mini8']
        prompt_sha=digest(json.dumps(inputs['inputs'][0].cpu().tolist()))
        if prompt_sha!=audit['prompt_sha256']:raise Blocked('Reference prompt differs from frozen audit')
        processor_sha=digest(json.dumps(self.processor.to_dict(),sort_keys=True))
        if processor_sha!=audit['processor_sha256'] or list(inputs['images'].shape[-2:])!=audit['processed_size']:
            raise Blocked('Reference image processing differs from frozen audit')
        evidence={'prefusion':{}};kv=[]
        def compress(module,args,output):evidence['visual_tokens']=int(output[0].shape[1])
        def prefusion(i):
            def capture(module,args,kwargs):
                hidden=args[0] if args else kwargs['hidden_states'];evidence['prefusion'][str(i)]=int(hidden.shape[1])
            return capture
        def cache(module,args,output):
            lengths=cache_lengths(output.past_key_values)
            if not kv:kv.append(lengths)
            elif len(kv)==1:kv.append(lengths)
            else:kv[-1]=lengths
        handles=[self.model.model.compressor.register_forward_hook(compress),self.model.register_forward_hook(cache)]
        handles += [layer.register_forward_pre_hook(prefusion(i),with_kwargs=True) for i,layer in enumerate(self.model.model.prefusion_layers)]
        try:result=generate_native(self.model,inputs,self.tokenizer,job,checkpoint)
        finally:
            for handle in handles:handle.remove()
        if evidence['visual_tokens']!=1 or evidence['prefusion']!=audit['prefusion_lengths']:
            raise Blocked('Actual Mini compression/prefusion differs from audit')
        if not kv or any(n!=audit['expanded_prompt_tokens'] for n in kv[0]):
            raise Blocked('Prefill cache differs from native audit')
        return dict(result,prompt_sha256=prompt_sha,prompt_ids=inputs['inputs'][0].cpu().tolist(),
            generation_eos_token_id=self.model.generation_config.eos_token_id,
            full_visual_tokens=1,actual_visual_tokens=1,
            prefusion_lengths=evidence['prefusion'],prefill_layer_kv=kv[0],final_layer_kv=kv[-1])


def prepare_jobs(spec,rows,identity,smoke=False):
    manifest_sha=identity.get('manifest_sha256')
    if not manifest_sha:raise Blocked('Frozen manifest SHA required')
    plan=build_natural(spec,rows,manifest_sha,include_qwen4=False)
    jobs=[j for j in plan['jobs'] if j['model']=='mini8' and j['side']=='reference']
    # A smoke is a bounded prefix of the exact calibration plan, not a new job
    # identity. Its immutable success may be reused by the full reference run.
    return jobs[:1] if smoke else jobs



def run_jobs(jobs,rows,out,session_factory,identity=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);by_uid={r['uid']:r for r in rows}
    session=None;count=0
    for job in jobs:
        row=by_uid[job['uid']];path=out/job['job_key'];path.mkdir(exist_ok=True)
        binding=dict(job=job,input_sha256=canonical(row),implementation_identity=identity or {});binding_path=path/'binding.json'
        if binding_path.exists():
            if read(binding_path)!=binding:raise Blocked('Calibration job binding changed')
        else:publish_once(binding_path,binding)
        result_path=path/'result.json'
        if result_path.exists():
            saved=read(result_path)
            if saved.get('binding_sha256')!=canonical(binding) or not saved.get('record_complete'):raise Blocked('Corrupt success receipt')
            count+=1;continue
        if list(path.glob('failure-*.json')):raise Blocked('Prior failure requires explicit reviewed recovery; no automatic reroll')
        try:
            if session is None:session=session_factory()
            write(out/'status.json',dict(stage='natural_calibration',completed=count,total=len(jobs),pid=os.getpid(),updated_at=time.time()))
            result=session.generate(row,job,path/'prefix.json')
            n=result['output_tokens'];cap=job['generation']['max_new_tokens']
            if n!=len(result['generated_token_ids']) or not 0<n<=cap:raise Blocked('Invalid generated length')
            if result['output_cap_reached']!=(n==cap):raise Blocked('Cap indicator differs')
            if not result['eos_terminated'] and n!=cap:raise Blocked('Premature stopping')
            publish_once(result_path,dict(job,**result,record_complete=True,binding_sha256=canonical(binding),
                scoring_status='pending_separate_label_process',implementation_identity=identity or {},native_gate_passed=False))
            count+=1
        except BaseException as exc:
            publish_once(path/('failure-'+str(time.time_ns())+'.json'),dict(error_type=type(exc).__name__,error=str(exc)[:1600],resampled=False))
            raise
    write(out/'completion.json',dict(stage='calibration_records_complete',completed=count,total=len(jobs),
        research_started=False,native_gate_passed=False,score_status='pending',
        scope='author-loader native HF calibration; shared fixed author model and sampler, not paged engine certification'))
    return count


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--numeric-receipt',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.manifest.name!='calibration_manifest.jsonl':raise Blocked('Frozen calibration only')
    proof=read(a.numeric_receipt)
    if proof.get('status')!='PASS' or proof.get('steps')!=1025 or not proof.get('kv_exact') or not all(proof.get('feature_exact',{}).values()) or len(proof.get('feature_exact',{}))!=5:
        raise Blocked('Mini independent integration diagnostic has not passed')
    rows=[json.loads(x) for x in a.manifest.read_text(encoding='utf-8').splitlines()]
    spec_path=Path(__file__).resolve().parent.parent/'round3_spec.json'
    identity=dict(manifest_sha256=digest(a.manifest.read_bytes()),spec_sha256=digest(spec_path.read_bytes()),
        code_sha256=code_hash(),numeric_receipt_sha256=digest(a.numeric_receipt.read_bytes()),
        source_lock_sha256=digest((a.folder/'manifests/source_lock.json').read_bytes()),
        weights_lock_sha256=digest((a.folder/'manifests/model_weights_lock.json').read_bytes()),
        reference_implementation='r3.mini_independent_probe.author_load/author_prepare',backend='author-native-hf-b1')
    jobs=prepare_jobs(read(spec_path),rows,identity,a.smoke)
    import fcntl
    with (a.folder/'correctness.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_jobs(jobs,rows,a.out,lambda:MiniAuthorSession(a.folder,a.out/('loading-'+str(time.time_ns()))),identity=identity)

if __name__=='__main__':main()
