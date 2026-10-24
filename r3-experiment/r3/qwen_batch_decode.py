"""Qwen-only mixed-length padded SDPA decode with native author B1 prefill.

This is a new numerical backend, NOT automatically certified. It owns no RNG,
EOS policy, output cap, scheduler, checkpoint, or result publication. The caller
samples prefill/decode logits and removes finished requests before the next call.
Padding is temporary per layer; stored KV tensors remain request-local/unpadded.
"""
import copy
import importlib
from .common import Blocked


def cache_pairs(cache):
    if hasattr(cache,'layers'):
        return [(layer.keys.detach(),layer.values.detach()) for layer in cache.layers]
    if hasattr(cache,'key_cache'):
        return [(k.detach(),v.detach()) for k,v in zip(cache.key_cache,cache.value_cache)]
    return [(entry[0].detach(),entry[1].detach()) for entry in cache]


def pad_layer(pairs,new_k,new_v):
    """Stage one appended token; never mutate source states or share storage."""
    import torch
    if new_k.ndim!=4 or new_k.shape!=new_v.shape or new_k.shape[2]!=1 or len(pairs)!=new_k.shape[0]:
        raise Blocked('Expected one new KV token per active request')
    updated=[]
    for index,(k,v) in enumerate(pairs):
        if k.shape!=v.shape or k.ndim!=4 or k.shape[0]!=1 or k.shape[1]!=new_k.shape[1] or k.shape[3]!=new_k.shape[3]:
            raise Blocked('Incompatible request cache layout')
        if k.device!=new_k.device or k.dtype!=new_k.dtype or v.dtype!=new_v.dtype:
            raise Blocked('Cache dtype/device mismatch')
        updated.append((torch.cat((k,new_k[index:index+1]),dim=2),
                        torch.cat((v,new_v[index:index+1]),dim=2)))
    lengths=[int(k.shape[2]) for k,_ in updated]
    shape=(len(updated),new_k.shape[1],max(lengths),new_k.shape[3])
    keys=new_k.new_zeros(shape);values=new_v.new_zeros(shape)
    for index,((k,v),length) in enumerate(zip(updated,lengths)):
        keys[index,:,:length]=k[0];values[index,:,:length]=v[0]
    valid=torch.arange(max(lengths),device=new_k.device)[None,:]<torch.tensor(lengths,device=new_k.device)[:,None]
    return keys,values,valid[:,None,None,:],updated


def padded_attention(query,keys,values,valid,scale):
    import torch.nn.functional as F
    # A single query attends all valid preceding/current keys, never padding.
    # is_causal=True would incorrectly mask a suffix query against leftmost KV.
    return F.scaled_dot_product_attention(query,keys,values,attn_mask=valid,
        dropout_p=0.,is_causal=False,scale=scale,enable_gqa=query.shape[1]!=keys.shape[1])


class Backend:
    backend_name='qwen-native-prefill-padded-sdpa-decode-v1'
    native_gate_passed=False

    def __init__(self,session):
        import torch
        if not getattr(session,'key','').startswith('qwen'):
            raise Blocked('Qwen decoder cannot accept LLaVA/Mini sessions')
        self.session=session;self.model=session.model;self.lm=self.model.model.language_model
        self.layers=self.lm.layers
        if getattr(self.lm.rotary_emb,'rope_type',None)!='default':
            raise Blocked('Dynamic/non-default RoPE needs separate batching certification')
        if next(self.model.parameters()).dtype!=torch.bfloat16:
            raise Blocked('Frozen batch backend requires BF16 model parameters')
        if any(not hasattr(layer.self_attn,'q_norm') or not hasattr(layer.self_attn,'k_norm') for layer in self.layers):
            raise Blocked('Expected native Qwen q/k normalization')
        module=importlib.import_module('transformers.models.qwen3_vl.modeling_qwen3_vl')
        self.apply_rope=module.apply_rotary_pos_emb
        self.context_bound=min(65536,int(self.lm.config.max_position_embeddings))

    def prefill(self,row,job):
        import torch
        inputs,metadata=self.session.prepare(row)
        if inputs['input_ids'].shape[0]!=1:raise Blocked('Author prefill must be B1')
        prompt_len=int(inputs['input_ids'].shape[1])
        if prompt_len+job['generation']['max_new_tokens']>self.context_bound:
            raise Blocked('Job exceeds native context bound')
        captured=[]
        def hook(module,args,output):
            captured.append((output.logits[:,-1,:].detach().float(),output.past_key_values))
        handle=self.model.register_forward_hook(hook)
        try:
            # One greedy dummy output is discarded. Scheduler receives the raw
            # logits and supplies the actual first token from its own uniform.
            with torch.inference_mode():
                self.model.generate(**inputs,max_new_tokens=1,do_sample=False,use_cache=True,
                    repetition_penalty=1.,min_new_tokens=0,return_dict_in_generate=False)
        finally:handle.remove()
        if len(captured)!=1:raise Blocked('Expected exactly one native prefill forward')
        logits,native_cache=captured[0]
        pairs=cache_pairs(native_cache)
        if len(pairs)!=len(self.layers) or any(k.shape[0]!=1 or k.shape!=v.shape for k,v in pairs):
            raise Blocked('Incomplete native per-layer KV')
        delta=self.model.model.rope_deltas
        if delta is None:raise Blocked('Native prefill did not produce rope_deltas')
        delta=delta.detach().clone()
        # Ask the native generation adapter for the NEXT token's actual four-
        # channel positions. Do not substitute compressed KV lengths for text
        # or multimodal positions. No second model forward is performed here.
        mask=inputs.get('attention_mask')
        if mask is None:mask=torch.ones_like(inputs['input_ids'])
        if not bool(mask.all()):raise Blocked('Single-request prefill unexpectedly padded')
        full_ids=torch.cat((inputs['input_ids'],inputs['input_ids'].new_zeros((1,1))),dim=1)
        full_mask=torch.cat((mask,mask.new_ones((1,1))),dim=1)
        with torch.inference_mode():
            prepared=self.model.prepare_inputs_for_generation(full_ids,past_key_values=native_cache,
                attention_mask=full_mask,cache_position=torch.tensor([prompt_len],device=full_ids.device),
                use_cache=True,is_first_iteration=False)
        positions=prepared.get('position_ids')
        if positions is None or tuple(positions.shape)!=(4,1,1):
            raise Blocked('Native Qwen next-position contract changed')
        state=dict(logits=logits,metadata=metadata,prefill_layer_kv=[int(k.shape[2]) for k,_ in pairs],
            cache=pairs,next_position_ids=positions.detach().clone(),rope_deltas=delta,
            native_rope_offset=int(getattr(self.model,'_rope_position_offset',0)),
            prompt_tokens=prompt_len,decode_steps=0,context_bound=self.context_bound,
            generation_eos_token_id=copy.deepcopy(self.model.generation_config.eos_token_id),
            owner=id(self),released=False,job_key=job['job_key'],
            position_source='native prepare_inputs_for_generation, then single-token +1')
        return state

    def decode(self,states,tokens):
        import torch
        if not states:raise Blocked('No active requests')
        if tokens.ndim!=1 or len(tokens)!=len(states) or tokens.dtype!=torch.long:
            raise Blocked('Expected sampled token IDs [batch] as torch.long')
        if len({id(s) for s in states})!=len(states):raise Blocked('Duplicate request in batch')
        for state in states:
            if state.get('owner')!=id(self) or state.get('released'):raise Blocked('Foreign/released request')
            if state['prompt_tokens']+state['decode_steps']+1>state['context_bound']:
                raise Blocked('Decode context exhausted')
        if tokens.device!=next(self.model.parameters()).device:raise Blocked('Token device mismatch')
        staged=[[] for _ in states]
        with torch.inference_mode():
            hidden=self.lm.embed_tokens(tokens[:,None])
            positions=torch.cat([s['next_position_ids'] for s in states],dim=1)
            # Native text model removes the leading text-position channel
            # before computing multimodal rotary embeddings.
            cos,sin=self.lm.rotary_emb(hidden,positions[1:])
            for i,layer in enumerate(self.layers):
                residual=hidden;x=layer.input_layernorm(hidden);attention=layer.self_attn
                shape=(len(states),1,-1,attention.head_dim)
                q=attention.q_norm(attention.q_proj(x).view(shape)).transpose(1,2)
                k=attention.k_norm(attention.k_proj(x).view(shape)).transpose(1,2)
                v=attention.v_proj(x).view(shape).transpose(1,2)
                q,k=self.apply_rope(q,k,cos,sin)
                padded_k,padded_v,valid,updated=pad_layer([s['cache'][i] for s in states],k,v)
                output=padded_attention(q,padded_k,padded_v,valid,attention.scaling)
                hidden=residual+attention.o_proj(output.transpose(1,2).reshape(len(states),1,-1).contiguous())
                hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
                for j,pair in enumerate(updated):staged[j].append(pair)
            logits=self.model.lm_head(self.lm.norm(hidden))[:,0].float()
        # Commit only after all layers succeed. An OOM/exception cannot leave
        # one request half advanced; scheduler must still preserve failure.
        for i,state in enumerate(states):
            state['cache']=staged[i];state['next_position_ids']=state['next_position_ids']+1
            state['decode_steps']+=1;state['logits']=logits[i:i+1]
            state['final_layer_kv']=self.kv_lengths(state)
        return logits

    @staticmethod
    def kv_lengths(state):
        return [int(k.shape[2]) for k,_ in state['cache']]

    @staticmethod
    def kv_bytes(state):
        return sum(t.numel()*t.element_size() for pair in state['cache'] for t in pair)

    def release(self,state):
        if state.get('owner')!=id(self):raise Blocked('Foreign request')
        state['cache']=[];state['logits']=None;state['released']=True
