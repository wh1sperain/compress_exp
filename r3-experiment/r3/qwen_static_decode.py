"""Candidate: append-only request KV, mask-free Flash SDPA, batched projections.
No production selection until numeric and representative long-cache timing pass.
"""
from .qwen_batch_decode import Backend as Original
from .common import Blocked

class Backend(Original):
    backend_name='qwen-static-request-kv-flash-v2-candidate'
    def prefill(self,row,job):
        import torch
        state=super().prefill(row,job)
        buffers=[];lengths=[]
        with torch.inference_mode():
            for k,v in state['cache']:
                n=k.shape[2];shape=(*k.shape[:2],n+job['generation']['max_new_tokens'],k.shape[3])
                kb=k.new_empty(shape);vb=v.new_empty(shape)
                kb[:,:,:n].copy_(k);vb[:,:,:n].copy_(v)
                buffers.append((kb,vb));lengths.append(n)
        state['buffers']=buffers;state['lengths']=lengths
        state['cache']=[(k[:,:,:n],v[:,:,:n]) for (k,v),n in zip(buffers,lengths)]
        return state

    def decode(self,states,tokens):
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel,SDPBackend
        if not states or tokens.ndim!=1 or len(tokens)!=len(states) or tokens.dtype!=torch.long:
            raise Blocked('Invalid token batch')
        if len({id(s) for s in states})!=len(states):raise Blocked('Duplicate state')
        for s in states:
            if s.get('owner')!=id(self) or s.get('released'):raise Blocked('Foreign state')
            if s['prompt_tokens']+s['decode_steps']+1>s['context_bound']:raise Blocked('Context exhausted')
            if any(n>=kv[0].shape[2] for n,kv in zip(s['lengths'],s['buffers'])):raise Blocked('KV capacity exceeded')
        with torch.inference_mode():
            hidden=self.lm.embed_tokens(tokens[:,None])
            positions=torch.cat([s['next_position_ids'] for s in states],dim=1)
            cos,sin=self.lm.rotary_emb(hidden,positions[1:])
            for i,layer in enumerate(self.layers):
                residual=hidden;x=layer.input_layernorm(hidden);a=layer.self_attn
                shape=(len(states),1,-1,a.head_dim)
                q=a.q_norm(a.q_proj(x).view(shape)).transpose(1,2)
                k=a.k_norm(a.k_proj(x).view(shape)).transpose(1,2)
                v=a.v_proj(x).view(shape).transpose(1,2)
                q,k=self.apply_rope(q,k,cos,sin)
                outputs=[]
                for j,s in enumerate(states):
                    kb,vb=s['buffers'][i];n=s['lengths'][i]
                    kb[:,:,n:n+1].copy_(k[j:j+1]);vb[:,:,n:n+1].copy_(v[j:j+1])
                    # No padding, no entire-cache copies, no explicit attention mask.
                    # Force Flash so unsupported shapes fail instead of silently using math.
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        outputs.append(F.scaled_dot_product_attention(q[j:j+1],kb[:,:,:n+1],vb[:,:,:n+1],
                            dropout_p=0.,is_causal=False,scale=a.scaling,enable_gqa=True))
                output=torch.cat(outputs,dim=0)
                hidden=residual+a.o_proj(output.transpose(1,2).reshape(len(states),1,-1).contiguous())
                hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
            logits=self.model.lm_head(self.lm.norm(hidden))[:,0].float()
        # Appends beyond committed lengths are invisible until all layers succeed.
        for j,s in enumerate(states):
            s['lengths']=[n+1 for n in s['lengths']]
            s['cache']=[(k[:,:,:n],v[:,:,:n]) for (k,v),n in zip(s['buffers'],s['lengths'])]
            s['next_position_ids']=s['next_position_ids']+1;s['decode_steps']+=1;s['logits']=logits[j:j+1]
            s['final_layer_kv']=self.kv_lengths(s)
        return logits

    @staticmethod
    def kv_bytes(state):
        return sum(t.numel()*t.element_size() for pair in state['buffers'] for t in pair)

    def release(self,state):
        super().release(state);state['buffers']=[]
