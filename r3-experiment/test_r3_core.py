import unittest
import numpy as np
from r3.scoring import extract,nls,r10
from r3.sampling import cpu_sample,uniforms
from r3.common import seed
from r3.build_jobs import build
from r3.manifest import TASKS,select,split_labels
from r3.rescue import masks
from r3.native_sources import visionzip_arguments
from r3.analyze import paired,intervals
from r3.common import Blocked
from r3.artifacts import verify_cached,sha256
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
from r3.compat import llava_forward_with_cache_position
from pathlib import Path
import json

class CoreTests(unittest.TestCase):
    def test_cache_position_bridge_preserves_arguments(self):
        calls=[];cache=object();position=object();embedding=object()
        class FakeLlama:
            @staticmethod
            def forward(self,**kw):calls.append(kw);return kw
        class FakeModel:
            def prepare_inputs_labels_for_multimodal(self,*args):
                self.prepared=args
                return None,position,'mask',cache,embedding,None
        model=FakeModel();cursor=object()
        with patch.dict('sys.modules',{'transformers':SimpleNamespace(LlamaForCausalLM=FakeLlama)}):
            result=llava_forward_with_cache_position(model,input_ids='ids',images='pixels',cache_position=cursor)
            self.assertIs(result['cache_position'],cursor)
            self.assertIs(result['past_key_values'],cache)
            self.assertIs(result['position_ids'],position)
            self.assertIs(result['inputs_embeds'],embedding)
            self.assertEqual(model.prepared[0],'ids')
            result=llava_forward_with_cache_position(model,inputs_embeds=embedding,cache_position=cursor)
            self.assertIs(result['inputs_embeds'],embedding)
    def test_cache_receipt_detects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.safetensors';p.write_bytes(b'valid-test-fixture')
            entry={'path':d,'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)}}}
            self.assertTrue(verify_cached(entry))
            p.write_bytes(b'changed-test-bytes')
            with self.assertRaises(Blocked):verify_cached(entry)
            with self.assertRaises(Blocked):verify_cached({'path':d,'files':{}})
    def test_repeat_pairing_precedes_bootstrap(self):
        rows=[]
        for repeat,delta in enumerate([0,300,600]):
            for condition,length in [('Full',100),('PriorTR25',100+delta)]:
                rows.append(dict(model='qwen8',stage='main',uid='u',task='DocVQA',image_sha256='i',
                                 repeat=repeat,condition=condition,seed=repeat,prompt_sha256='p',
                                 reasoning_tokens=length,output_cap_reached=False,score=1))
        value=paired(rows,'qwen8','PriorTR25','reasoning_tokens')[('DocVQA','i')]
        self.assertEqual(value[0],300)
        self.assertAlmostEqual(value[1],2/3)
        with self.assertRaises(Blocked):paired(rows[:-1],'qwen8','PriorTR25','reasoning_tokens')
        with self.assertRaises(Blocked):paired(rows+[rows[0]],'qwen8','PriorTR25','reasoning_tokens')
    def test_visionzip_public_budget(self):
        for k,d,c in [(288,243,45),(144,122,22)]:
            self.assertEqual(visionzip_arguments(k),dict(dominant=d,contextual=c))
            self.assertEqual(d+c,k)
        with self.assertRaises(ValueError):visionzip_arguments(576)
    def test_required_scores(self):
        for a,b,x in [('11 a.m.','11 a.m.',1),('3 Million Users','3 million',.6),('34 Million','3 million',.9)]:
            p,_=extract('FINAL ANSWER: '+a);self.assertAlmostEqual(nls(p,[b]),x)
    def test_last_marker_and_units(self):
        self.assertEqual(extract('FINAL ANSWER: 2\nFINAL ANSWER: $32%.')[0],'$32%.')
        self.assertIsNone(extract('I guess 42')[0])
        self.assertIsNone(extract(r'\boxed{42')[0])
        self.assertEqual(extract(r'\boxed{a{b}}')[0],'a{b}')
    def test_ties_and_stream(self):
        self.assertEqual(cpu_sample([0,0,0],0,top_k=2),0)
        self.assertEqual(cpu_sample([0,0,0],.75,top_k=2),1)
        np.testing.assert_equal(uniforms(seed('x',0),10)[5:],uniforms(seed('x',0),10)[5:])
        self.assertNotEqual(seed('x',0),seed('x',1))
    def test_repetition(self):
        self.assertEqual(r10('short text'),0)
        self.assertGreater(r10('hello '*100),.5)
    def test_jobs_full_registration(self):
        spec=json.loads(Path('round3_spec.json').read_text())
        rows=[{'uid':f'{t}:{i}','task':t,'image_sha256':f'{t}-image-{i}'} for t in TASKS for i in range(100)]
        mechanism=[r for r in rows if r['task'] in TASKS[:2] and int(r['uid'].split(':')[-1])<30]
        jobs,counts=build(spec,rows,mechanism,'test-profile')
        self.assertEqual(len(jobs),23160)
        self.assertEqual(counts,dict(main=19500,controls=2400,rescue=720,stop=540))
        self.assertTrue(all('answer' not in j for j in jobs))
    def test_rescue_matched_controls(self):
        result=masks('u',[0,1,2,3],[4],[5],list(range(10)))
        for label in ('Value','Binding'):
            self.assertEqual(result[label+'Rescue']['removed'],result[label+'Random']['removed'])
            self.assertEqual(len(result[label+'Rescue']['selected_indices']),4)
            self.assertFalse(set(result[label+'Random']['added'])&{4,5})
        self.assertEqual(result,masks('u',[0,1,2,3],[4],[5],list(range(10))))
if __name__=='__main__':unittest.main()
