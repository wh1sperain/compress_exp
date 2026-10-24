"""Native per-request LLaVA prefill, actual multi-request SDPA decode.

Preserves each layer's author KV length, including FastV's longer first layers.
This is padded dense batch attention, not paged attention or a certification.
The scheduler owns all research RNG, sampled tokens, stopping and checkpoints.
"""
import importlib
import inspect
import math
from .common import Blocked


def layer_layout(lengths):
    if not lengths or any(type(n) is not int or n<=0 for n in lengths):
        raise Blocked('Positive per-request layer lengths required')
    maximum=max(lengths)
    return {'width':maximum+1,'valid_lengths':[n+1 for n in lengths],
            'right_padding':[maximum-n for n in lengths],
            'attention_mask':[[True]*(n+1)+[False]*(maximum-n) for n in lengths]}


def legacy_cache(cache):
    if hasattr(cache,'to_legacy_cache'):cache=cache.to_legacy_cache()
    if not isinstance(cache,(tuple,list)) or not cache:raise Blocked('Native prefill cache absent')
    result=[]
    for pair in cache:
        if len(pair)!=2 or pair[0].ndim!=4 or pair[0].shape!=pair[1].shape or pair[0].shape[0]!=1:
            raise Blocked('Unexpected B1 native KV layout')
        result.append((pair[0].detach(),pair[1].detach()))
    return result


class Backend:
    def __init__(self,session):
        self.session=session;self.model=session.model;self.decoder=self.model.model
        if getattr(session,'key',None)!='llava7':raise Blocked('LLaVA backend only; Mini has its own native backend')
        if getattr(self.model.config,'pretraining_tp',1)!=1:raise Blocked('Tensor-parallel author projection path unsupported')
        self.layers=self.decoder.layers
        self.rotary=[]
        for layer in self.layers:
            attention=layer.self_attn
            module=importlib.import_module(type(attention).__module__)
            if not hasattr(module,'apply_rotary_pos_emb'):raise Blocked('Author rotary function unavailable')
            positional='position_ids' in inspect.signature(attention.rotary_emb.forward).parameters
            self.rotary.append((module.apply_rotary_pos_emb,positional))

    def prefill(self,row,job):
        import torch
        from transformers import LogitsProcessorList
        from .model_resume_probe import diagnostic_no_eos
        inputs,metadata=self.session.prepare(row)
        captured={};handles=[]
        class PositionCaptured(Exception):pass
        def output_hook(module,args,output):
            if 'cache' in captured:raise Blocked('Unexpected extra forward during prefill position probe')
            captured['cache']=legacy_cache(output.past_key_values)
            captured['logits']=output.logits[:,-1,:].detach()
        def first_layer(module,args,kwargs):
            if 'cache' not in captured:return
            position=kwargs.get('position_ids')
            if position is None or position.numel()!=1:raise Blocked('Actual native next-token position unavailable')
            captured['next_position']=int(position.item())
            cache_position=kwargs.get('cache_position')
            captured['cache_position']=None if cache_position is None else cache_position.detach().cpu().tolist()
            # Entered the next native layer, but no dummy-token layer computation
            # occurs. The original prefill cache is untouched.
            raise PositionCaptured()
        class Dummy:
            def __call__(self,input_ids,scores):
                # No sampling/uniform stream: this token only drives native
                # prepare_inputs_for_generation into its position calculation.
                forced=torch.full_like(scores,-float('inf'));forced[:,0]=0
                return forced
        handles.append(self.model.register_forward_hook(output_hook))
        handles.append(self.layers[0].register_forward_pre_hook(first_layer,with_kwargs=True))
        try:
            with torch.inference_mode(),diagnostic_no_eos(self.model) as config:
                self.model.generate(**inputs,generation_config=config,max_new_tokens=2,do_sample=False,
                    use_cache=True,eos_token_id=None,forced_eos_token_id=None,
                    logits_processor=LogitsProcessorList([Dummy()]))
        except PositionCaptured:pass
        finally:
            for handle in handles:handle.remove()
        if not {'cache','logits','next_position'}<=captured.keys():raise Blocked('Native next-position probe did not complete')
        if len(captured['cache'])!=len(self.layers):raise Blocked('Incomplete author cache')
        lengths=[int(pair[0].shape[-2]) for pair in captured['cache']]
        metadata=dict(metadata,batch_backend='llava-native-prefill-dense-sdpa-v1',
            native_next_position=captured['next_position'],native_next_cache_position=captured['cache_position'],
            position_probe_dummy_tokens_in_research=0)
        if self.session.condition=='Random25_L2' and job.get('config',{}).get('mask_seed')!=metadata['control_evidence']['mask_seed']:
            raise Blocked('Random control differs from registered mask seed')
        return {'logits':captured['logits'],'prefill_layer_kv':lengths,'metadata':metadata,
                'opaque':{'cache':captured['cache'],'next_position':captured['next_position'],
                          'backend_id':id(self),'decoded_steps':0}}

    def kv_lengths(self,state):
        return [int(k.shape[-2]) for k,v in state['opaque']['cache']]

    def kv_bytes(self,state):
        return sum(t.numel()*t.element_size() for pair in state['opaque']['cache'] for t in pair)

    def reserved_kv_bytes(self,state,remaining_tokens):
        if type(remaining_tokens) is not int or remaining_tokens<0:raise Blocked('Invalid remaining output reservation')
        return sum((k.shape[-2]+remaining_tokens)*k.shape[0]*k.shape[1]*k.shape[3]*k.element_size()*2
                   for k,v in state['opaque']['cache'])

    def decode(self,states,tokens):
        import torch
        import torch.nn.functional as F
        if not states or len({id(s) for s in states})!=len(states):raise Blocked('Empty/duplicate live request batch')
        if tokens.ndim!=1 or tokens.shape[0]!=len(states):raise Blocked('One original sampled token per live request required')
        if any(s['opaque'].get('backend_id')!=id(self) for s in states):raise Blocked('Different model/condition cache cannot share this batch')
        if any(len(s['opaque']['cache'])!=len(self.layers) for s in states):raise Blocked('Incomplete per-layer cache')
        if any(s['opaque'].get('decode_in_progress') for s in states):raise Blocked('Failed partial decode requires original-prefix reconstruction')
        for state in states:state['opaque']['decode_in_progress']=True
        with torch.inference_mode():
            positions=torch.tensor([[s['opaque']['next_position']] for s in states],device=tokens.device,dtype=torch.long)
            hidden=self.decoder.embed_tokens(tokens[:,None]);batch=len(states)
            for index,layer in enumerate(self.layers):
                attention=layer.self_attn;residual=hidden
                normalized=layer.input_layernorm(hidden)
                heads=attention.num_heads;kv_heads=attention.num_key_value_heads;dim=attention.head_dim
                query=attention.q_proj(normalized).view(batch,1,heads,dim).transpose(1,2)
                key=attention.k_proj(normalized).view(batch,1,kv_heads,dim).transpose(1,2)
                value=attention.v_proj(normalized).view(batch,1,kv_heads,dim).transpose(1,2)
                lengths=[int(s['opaque']['cache'][index][0].shape[-2]) for s in states]
                rotate,positional=self.rotary[index]
                if positional:cos,sin=attention.rotary_emb(value,positions)
                else:cos,sin=attention.rotary_emb(value,seq_len=max(max(lengths)+1,max(s['opaque']['next_position'] for s in states)+1))
                query,key=rotate(query,key,cos,sin,positions)
                width=max(lengths)+1
                keys=torch.zeros((batch,kv_heads,width,dim),dtype=key.dtype,device=key.device)
                values=torch.zeros_like(keys)
                for slot,state in enumerate(states):
                    old_key,old_value=state['opaque']['cache'][index]
                    new_key=torch.cat((old_key,key[slot:slot+1]),dim=-2)
                    new_value=torch.cat((old_value,value[slot:slot+1]),dim=-2)
                    state['opaque']['cache'][index]=(new_key,new_value)
                    n=lengths[slot]+1;keys[slot,:,:n]=new_key[0];values[slot,:,:n]=new_value[0]
                if heads!=kv_heads:
                    if heads%kv_heads:raise Blocked('Invalid native grouped attention layout')
                    keys=keys.repeat_interleave(heads//kv_heads,dim=1);values=values.repeat_interleave(heads//kv_heads,dim=1)
                valid=torch.tensor([n+1 for n in lengths],device=keys.device)
                mask=torch.arange(width,device=keys.device)[None,:]<valid[:,None]
                attended=F.scaled_dot_product_attention(query,keys,values,attn_mask=mask[:,None,None,:],
                    dropout_p=0.,is_causal=False,scale=1/math.sqrt(dim))
                hidden=residual+attention.o_proj(attended.transpose(1,2).contiguous().reshape(batch,1,heads*dim))
                hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
            hidden=self.decoder.norm(hidden)
            logits=self.model.lm_head(hidden)[:,0,:].float()
            for state in states:
                state['opaque']['next_position']+=1;state['opaque']['decoded_steps']+=1
                state['opaque']['decode_in_progress']=False
            return logits

    def release(self,state):
        state['opaque']['cache']=[]
        state.pop('logits',None)
