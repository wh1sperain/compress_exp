import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from r3.batch_scheduler import Scheduler, reserve_bytes
from r3.common import Blocked, write, read

class FakeBackend:
    def __init__(self):self.states=[];self.batches=[];self.released=[]
    def prefill(self,row,job):
        s={'logits':torch.tensor([[0.,1.,2.]],dtype=torch.float64),'metadata':{},'prefill_layer_kv':[row['position']], 'position':row['position']}
        self.states.append(s);return s
    def decode(self,states,tokens):
        self.batches.append([(s['position'],int(t)) for s,t in zip(states,tokens)])
        for s in states:s['position']+=1
        return torch.cat([s['logits'] for s in states])
    def kv_lengths(self,state):return [state['position']]
    def release(self,state):
        assert id(state) not in self.released,'double release'
        self.released.append(id(state))

class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.backend=FakeBackend()
        config=NS(num_hidden_layers=2,num_key_value_heads=1,num_attention_heads=1,hidden_size=2)
        self.session=NS(key='llava7',model=NS(config=config,device='cpu',generation_config=NS(eos_token_id=2)),tokenizer=NS(decode=lambda ids,**kw:str(ids)))
        with patch('r3.batch_scheduler.backend_for',return_value=self.backend),patch('torch.cuda.mem_get_info',return_value=(20*1024**3,24*1024**3)),patch('torch.cuda.memory_reserved',return_value=0):
            self.scheduler=Scheduler(self.session,dict(max_live_requests=2,dtype='bfloat16',kv_dtype='bfloat16',backend='fake'))
    def tearDown(self):self.tmp.cleanup()
    def request(self,name,cap=3,seed=0,temp=1,position=5):
        row={'position':position,'native_inputs':{'llava7':{'expanded_prompt_tokens':position}}}
        job={'job_key':name,'seed':seed,'generation':dict(max_new_tokens=cap,temperature=temp,top_k=3,top_p=1.)}
        return row,job,self.root/(name+'.json')
    def run_requests(self,requests,sample):
        with patch('r3.batch_scheduler.gpu_sample',side_effect=sample):return list(self.scheduler.run(requests,self.root/'status.json'))
    def test_slots_independent_rng_parameters_positions_and_cap(self):
        seen=[]
        def sample(logits,u,**kw):seen.append((u.tolist(),kw['temperature']));return torch.zeros(len(u),dtype=torch.long)
        requests=[self.request('a',1,7),self.request('b',3,8,position=11),self.request('c',2,9,temp=.6,position=21)]
        out=self.run_requests(requests,sample)
        self.assertEqual([x[0]['job_key'] for x in out],['a','b','c'])
        self.assertEqual([x[2]['output_tokens'] for x in out],[1,3,2])
        self.assertEqual(self.backend.batches,[[(11,0)],[(12,0),(21,0)]])
        from r3.sampling import uniforms
        flat=[v for values,t in seen for v in values]
        for seed,cap in [(7,1),(8,3),(9,2)]:
            for v in uniforms(seed,cap):self.assertIn(float(v),flat)
        self.assertIn(.6,[t for _,t in seen]);self.assertEqual(len(self.backend.released),3)
    def test_eos_and_generator_close_release_once(self):
        def sample(logits,u,**kw):return torch.tensor([2,0][:len(u)])
        with patch('r3.batch_scheduler.gpu_sample',side_effect=sample):
            gen=self.scheduler.run([self.request('a'),self.request('b')],self.root/'s.json')
            result=next(gen)[2];self.assertTrue(result['eos_terminated']);gen.close()
        self.assertEqual(len(self.backend.released),2)
        self.assertEqual(read(self.root/'b.json')['generated_token_ids'],[0])
    def test_prefix_replay_preserves_longer_on_mismatch(self):
        request=self.request('a');write(request[2],dict(job_key='a',seed=0,generated_steps=2,generated_token_ids=[0,1]))
        with self.assertRaisesRegex(Blocked,'prefix reconstruction'):
            self.run_requests([request],lambda logits,u,**kw:torch.zeros(len(u),dtype=torch.long))
        self.assertEqual(read(request[2])['generated_token_ids'],[0,1]);self.assertEqual(len(self.backend.released),1)
    def test_prefix_replay_exact(self):
        request=self.request('a');write(request[2],dict(job_key='a',seed=0,generated_steps=2,generated_token_ids=[0,0]))
        result=self.run_requests([request],lambda logits,u,**kw:torch.zeros(len(u),dtype=torch.long))[0][2]
        self.assertEqual(result['reconstructed_prefix_tokens'],2);self.assertEqual(result['generated_token_ids'],[0,0,0])
    def test_reservation_serializes_and_rejects(self):
        self.scheduler.kv_budget=reserve_bytes(self.session.model.config,5,3)
        self.run_requests([self.request('a'),self.request('b')],lambda logits,u,**kw:torch.zeros(len(u),dtype=torch.long))
        self.assertTrue(all(len(batch)==1 for batch in self.backend.batches))
        self.scheduler.kv_budget=1
        with self.assertRaisesRegex(Blocked,'exceeds'):self.run_requests([self.request('c')],None)
    def test_cleanup_failure_does_not_hide_decode_failure(self):
        calls=[]
        def release(state):
            calls.append(id(state))
            raise RuntimeError('release failed')
        self.backend.release=release
        self.backend.decode=lambda *args:(_ for _ in ()).throw(ValueError('original decode'))
        with self.assertRaisesRegex(ValueError,'original decode'):
            self.run_requests([self.request('a'),self.request('b')],lambda logits,u,**kw:torch.zeros(len(u),dtype=torch.long))
        self.assertEqual(len(calls),2)

    def test_decode_failure_preserves_and_releases(self):
        self.backend.decode=lambda *args:(_ for _ in ()).throw(RuntimeError('decode failure'))
        with self.assertRaisesRegex(RuntimeError,'decode failure'):
            self.run_requests([self.request('a'),self.request('b')],lambda logits,u,**kw:torch.zeros(len(u),dtype=torch.long))
        self.assertEqual(len(self.backend.released),2)
        self.assertEqual(read(self.root/'a.json')['generated_token_ids'],[0])

if __name__=='__main__':unittest.main()
