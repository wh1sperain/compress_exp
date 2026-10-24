"""R3 candidate adapted from Compress/src/methods_qwen3.py PagedDecoder.
Reuses BF16 physical pages, FlashInfer decode and fixed-shape CUDA graph.
Native R3 prefill and per-request original mRoPE positions remain authoritative.
No sampler, scientific conditions or stopping policies are imported from R2.
"""
import math
from .qwen_batch_decode import Backend as Native
from .common import Blocked

class Backend(Native):
    backend_name='qwen-flashinfer-paged-graph-r2-adapted-candidate'
    def __init__(self,session):
        super().__init__(session)
        import torch,flashinfer
        # Author vision code keeps its explicit attention-weight path whenever
        # weights are requested. Only ordinary vision attention uses SDPA.
        # Independent eager-reference logits must validate this candidate.
        self.model.model.visual.config._attn_implementation='sdpa'
        c=self.lm.config
        self.batch_size=int(getattr(session,'paged_batch_size',8));self.page_size=256
        self.heads=c.num_key_value_heads;self.dim=c.head_dim;self.q_heads=c.num_attention_heads
        # Preserve workspace for unchanged high-resolution eager vision prefill.
        gib=float(getattr(session,'paged_cache_gib',24))
        self.pages=int(gib*1024**3)//(len(self.layers)*2*self.page_size*self.heads*self.dim*2)
        self.kv=torch.zeros((len(self.layers),self.pages,2,self.page_size,self.heads,self.dim),device=self.model.device,dtype=torch.bfloat16)
        self.free=list(range(self.batch_size,self.pages));self.live=set()
        self.workspace=torch.empty(128*1024**2,dtype=torch.uint8,device=self.model.device)
        self.indptr=torch.zeros(self.batch_size+1,dtype=torch.int32,device=self.model.device)
        self.indices=torch.zeros(self.pages,dtype=torch.int32,device=self.model.device)
        self.last=torch.ones(self.batch_size,dtype=torch.int32,device=self.model.device)
        self.wrapper=flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace,'NHD',use_cuda_graph=True,use_tensor_cores=True,
            paged_kv_indptr_buffer=self.indptr,paged_kv_indices_buffer=self.indices,paged_kv_last_page_len_buffer=self.last)
        self.ids=torch.zeros((self.batch_size,1),dtype=torch.long,device=self.model.device)
        self.positions=torch.zeros((3,self.batch_size,1),dtype=torch.long,device=self.model.device)
        self.write_pages=torch.zeros(self.batch_size,dtype=torch.long,device=self.model.device)
        self.write_offsets=torch.zeros(self.batch_size,dtype=torch.long,device=self.model.device)
        self.graph=None

    def prefill(self,row,job):
        import torch
        state=super().prefill(row,job)
        lengths=self.kv_lengths(state)
        if len(set(lengths))!=1:raise Blocked('Paged Qwen requires verified equal layer lengths')
        n=lengths[0];reserve=math.ceil((n+job['generation']['max_new_tokens'])/self.page_size)
        # Reserve full remaining cap; no overcommit or CPU swap in this candidate.
        if reserve>len(self.free):raise Blocked('Full-cap page admission exhausted')
        pages=[self.free.pop() for _ in range(reserve)]
        with torch.inference_mode():
            for i,(k,v) in enumerate(state['cache']):
                for j in range(math.ceil(n/self.page_size)):
                    size=min(self.page_size,n-j*self.page_size);start=j*self.page_size
                    self.kv[i,pages[j],0,:size].copy_(k[0,:,start:start+size].transpose(0,1))
                    self.kv[i,pages[j],1,:size].copy_(v[0,:,start:start+size].transpose(0,1))
        state['pages']=pages;state['length']=n;state['cache']=[];self.live.add(id(state))
        return state

    def _forward(self):
        hidden=self.lm.embed_tokens(self.ids);cos,sin=self.lm.rotary_emb(hidden,self.positions)
        for i,layer in enumerate(self.layers):
            residual=hidden;x=layer.input_layernorm(hidden);a=layer.self_attn
            shape=(self.batch_size,1,-1,self.dim)
            q=a.q_norm(a.q_proj(x).view(shape)).transpose(1,2)
            k=a.k_norm(a.k_proj(x).view(shape)).transpose(1,2)
            v=a.v_proj(x).view(shape)[:,0]
            q,k=self.apply_rope(q,k,cos,sin)
            self.kv[i,self.write_pages,0,self.write_offsets]=k[:,:,0]
            self.kv[i,self.write_pages,1,self.write_offsets]=v
            output=self.wrapper.run(q[:,:,0].contiguous(),self.kv[i])
            hidden=residual+a.o_proj(output.reshape(self.batch_size,1,-1))
            hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
        return self.model.lm_head(self.lm.norm(hidden))[:,0].float()

    def decode(self,states,tokens):
        import torch
        if not states or len(states)>self.batch_size or tokens.shape!=(len(states),) or tokens.dtype!=torch.long:raise Blocked('Invalid paged batch')
        if len({id(s) for s in states})!=len(states):raise Blocked('Duplicate request')
        ptr=[0];indices=[];last=[];wp=[];wo=[]
        self.ids.zero_();self.positions.zero_();self.ids[:len(states),0].copy_(tokens)
        for j in range(self.batch_size):
            if j<len(states):
                s=states[j]
                if s.get('owner')!=id(self) or id(s) not in self.live or s.get('released'):raise Blocked('Foreign state')
                if s['prompt_tokens']+s['decode_steps']+1>s['context_bound']:raise Blocked('Context exhausted')
                n=s['length'];count=math.ceil((n+1)/self.page_size)
                if count>len(s['pages']):raise Blocked('Reserved pages exhausted')
                pages=s['pages'][:count];self.positions[:,j:j+1].copy_(s['next_position_ids'][1:])
            else:n=0;pages=[j]
            indices.extend(pages);ptr.append(len(indices));last.append(n%self.page_size+1)
            wp.append(pages[n//self.page_size]);wo.append(n%self.page_size)
        self.wrapper.plan(torch.tensor(ptr,dtype=torch.int32),torch.tensor(indices,dtype=torch.int32),torch.tensor(last,dtype=torch.int32),
            self.q_heads,self.heads,self.dim,self.page_size,pos_encoding_mode='NONE',q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,non_blocking=False)
        self.write_pages.copy_(torch.tensor(wp,dtype=torch.long));self.write_offsets.copy_(torch.tensor(wo,dtype=torch.long))
        with torch.inference_mode():
            if self.graph is None:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):self._forward()
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                self.graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):self.graph_logits=self._forward()
            self.graph.replay()
            # Copy outputs: a subsequent graph must not mutate another caller's saved logits.
            logits=self.graph_logits[:len(states)].clone()
        for j,s in enumerate(states):
            s['length']+=1;s['decode_steps']+=1;s['next_position_ids']=s['next_position_ids']+1;s['logits']=logits[j:j+1]
            s['final_layer_kv']=self.kv_lengths(s)
        return logits

    def kv_lengths(self,state):
        if 'length' not in state:return super().kv_lengths(state)
        return [state['length']]*len(self.layers)

    def kv_bytes(self,state):
        return len(state['pages'])*len(self.layers)*2*self.page_size*self.heads*self.dim*2

    def release(self,state):
        if state.get('owner')!=id(self) or id(state) not in self.live:raise Blocked('Foreign or released state')
        self.free.extend(state['pages']);self.live.remove(id(state));state['pages']=[]
        super().release(state)
