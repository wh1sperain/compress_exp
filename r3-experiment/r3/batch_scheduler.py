"""One resident native model, bounded live requests, independent RNG and KV.

The backend alone owns native prefill/positions; this scheduler never scores or
parses an answer. A dense padded backend reserves scratch as well as final KV.
"""
import time
import sys
from pathlib import Path
from .common import Blocked, read, write
from .sampling import uniforms, gpu_sample
from .prefix_checkpoint import publish


def note_error(error, message):
    if hasattr(error, 'add_note'):error.add_note(message)
    else:error.__dict__.setdefault('__notes__', []).append(message)


def backend_for(session):
    if session.key.startswith('qwen'):
        from .qwen_batch_decode import Backend
    elif session.key=='mini8':
        from .mini_batch_decode import Backend
    else:
        from .llava_batch_decode import Backend
    return Backend(session)


def reserve_bytes(config, prompt, cap, scratch_factor=3):
    if hasattr(config,'text_config'): config=config.text_config
    layers=config.num_hidden_layers
    heads=getattr(config,'num_key_value_heads',config.num_attention_heads)
    dim=getattr(config,'head_dim',config.hidden_size//config.num_attention_heads)
    return int(2*2*layers*heads*dim*(prompt+cap)*scratch_factor)


class Scheduler:
    def __init__(self, session, profile):
        import torch
        self.session=session;self.model=session.model;self.backend=backend_for(session)
        self.profile=profile;self.maximum=int(profile['max_live_requests'])
        if not 1<=self.maximum<=32:raise Blocked('Unapproved live slot count')
        if profile.get('dtype')!='bfloat16' or profile.get('kv_dtype')!='bfloat16':raise Blocked('Unapproved precision')
        self.scratch=float(profile.get('kv_scratch_factor',3))
        if self.scratch<3:raise Blocked('Dense batch scratch reservation must be validated before reduction')
        free,total=torch.cuda.mem_get_info()
        self.kv_budget=int(min(free-8*1024**3,total*.90-torch.cuda.memory_reserved()))
        if self.kv_budget<=0:raise Blocked('Insufficient conservative live KV budget')
        self.device=self.model.device
        eos=self.model.generation_config.eos_token_id
        self.eos=set([] if eos is None else [eos] if isinstance(eos,int) else eos)
        if not self.eos:raise Blocked('Native EOS configuration missing')

    def run(self, requests, status_path):
        """Yield (original job, original row, complete raw generation) per EOS/cap.

        requests is a lazy iterator of (row, job, original checkpoint path).
        Every original prefix must reconstruct before a result can be committed.
        """
        import torch
        pending=iter(requests);next_request=None;exhausted=False;active=[];completed=0
        peak_live=0;steps=0;started=time.time();last_status=0
        def flush(request):
            receipt={'job_key':request['job']['job_key'],'seed':request['job']['seed'],
                     'generated_token_ids':request['ids'],'generated_steps':len(request['ids']),
                     'updated_at':time.time(),'rebuild_prefix_tokens':len(request['saved'])}
            publish(request['checkpoint'],receipt,request['saved'])
        def release(request):
            if not request.get('released', False):
                request['released'] = True
                self.backend.release(request['state'])
        try:
            while active or not exhausted:
                while len(active)<self.maximum and not exhausted:
                    if next_request is None:
                        try: next_request=next(pending)
                        except StopIteration: exhausted=True;break
                    row,job,checkpoint=next_request
                    audit=row['native_inputs'][self.session.key]
                    reservation=reserve_bytes(self.model.config,audit['expanded_prompt_tokens'],job['generation']['max_new_tokens'],self.scratch)
                    if reservation>self.kv_budget:raise Blocked('One full-cap request exceeds verified KV admission budget')
                    if sum(r['reservation'] for r in active)+reservation>self.kv_budget:break
                    checkpoint=Path(checkpoint);saved=[]
                    if checkpoint.exists():
                        previous=read(checkpoint)
                        if previous.get('job_key')!=job['job_key'] or previous.get('seed')!=job['seed'] or previous.get('generated_steps')!=len(previous['generated_token_ids']):
                            raise Blocked('Original prefix identity/RNG mismatch')
                        saved=previous['generated_token_ids']
                    if len(saved)>job['generation']['max_new_tokens']:raise Blocked('Original prefix exceeds output cap')
                    u=torch.tensor(uniforms(job['seed'],job['generation']['max_new_tokens']),device=self.device)
                    state=self.backend.prefill(row,job)
                    active.append(dict(row=row,job=job,checkpoint=checkpoint,saved=saved,ids=[],state=state,
                        reservation=reservation,u=u,start=time.time()))
                    next_request=None;peak_live=max(peak_live,len(active))
                if not active:
                    if exhausted:break
                    raise Blocked('Admission made no progress')
                # Sampling parameters belong to each request; KV decode remains batched.
                groups={}
                for index,r in enumerate(active):
                    g=r['job']['generation']
                    key=(g['temperature'],g['top_k'],g['top_p'])
                    groups.setdefault(key,[]).append(index)
                token_ids=[None]*len(active)
                for (temperature,top_k,top_p),indices in groups.items():
                    logits=torch.cat([active[i]['state']['logits'] for i in indices],dim=0)
                    u=torch.stack([active[i]['u'][len(active[i]['ids'])] for i in indices])
                    sampled=gpu_sample(logits,u,temperature=temperature,top_k=top_k,top_p=top_p)
                    for i,token in zip(indices,sampled.tolist()):token_ids[i]=token
                survivors=[];survivor_tokens=[];ready=[]
                for r,token in zip(active,token_ids):
                    index=len(r['ids'])
                    if index<len(r['saved']) and token!=r['saved'][index]:raise Blocked('Batch prefix reconstruction differs; no resampling')
                    r['ids'].append(token);steps+=1
                    cap=r['job']['generation']['max_new_tokens'];ended=token in self.eos
                    if len(r['ids'])%128==0 or ended or len(r['ids'])==cap:flush(r)
                    if ended or len(r['ids'])==cap:
                        if len(r['ids'])<len(r['saved']):raise Blocked('Ended before original prefix reconstructed')
                        state=r['state'];ids=r['ids']
                        result=dict(state['metadata'],generated_token_ids=ids,output_tokens=len(ids),
                            output_text=self.session.tokenizer.decode(ids,skip_special_tokens=False),
                            eos_terminated=ended,output_cap_reached=len(ids)==cap,
                            prefill_layer_kv=state['prefill_layer_kv'],final_layer_kv=self.backend.kv_lengths(state),
                            generation_eos_token_id=self.model.generation_config.eos_token_id,
                            wall_seconds=time.time()-r['start'],reconstructed_prefix_tokens=len(r['saved']),
                            batch_backend=self.profile['backend'],live_limit=self.maximum)
                        release(r);completed+=1
                        ready.append((r['job'],r['row'],result))
                    else:survivors.append(r);survivor_tokens.append(token)
                active=survivors
                # Transfer ownership before yielding: generator close must not release
                # completed states twice, nor lose the remaining live requests.
                for item in ready:yield item
                if active:
                    outputs=self.backend.decode([r['state'] for r in active],torch.tensor(survivor_tokens,device=self.device,dtype=torch.long))
                    if outputs.shape[0]!=len(active):raise Blocked('Batch result row count mismatch')
                    for index,r in enumerate(active):r['state']['logits']=outputs[index:index+1]
                if time.time()-last_status>=10:
                    write(status_path,dict(stage='generating',completed=completed,active=len(active),peak_live=peak_live,
                        active_tokens={r['job']['job_key']:len(r['ids']) for r in active},generated_steps=steps,
                        kv_budget=self.kv_budget,reserved_bytes=sum(r['reservation'] for r in active),updated_at=time.time(),elapsed_seconds=time.time()-started))
                    last_status=time.time()
            write(status_path,dict(stage='complete',completed=completed,active=0,peak_live=peak_live,
                generated_steps=steps,updated_at=time.time(),elapsed_seconds=time.time()-started))
        except BaseException as error:
            for r in active:
                # Persist each original sampled prefix on infrastructure failure.
                # A caller records failures; it must never silently rerun these jobs.
                try:flush(r)
                except Exception as checkpoint_error:
                    note_error(error,'Checkpoint preservation failed: '+repr(checkpoint_error))
            raise
        finally:
            original_error=sys.exc_info()[1]
            cleanup_errors=[]
            for r in active:
                try:release(r)
                except Exception as cleanup_error:cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                if original_error is not None:
                    for error in cleanup_errors:note_error(original_error,'Backend cleanup failed: '+repr(error))
                else:raise cleanup_errors[0]
