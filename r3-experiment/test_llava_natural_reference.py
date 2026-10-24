import ast
import unittest
from pathlib import Path
from r3.common import Blocked,read
from r3.manifest import TASKS
from r3.natural_preparation import build
from r3.llava_natural_reference import transform_builder,guard_resize,prepare_jobs,CONDITIONS

class LlavaNaturalReferenceTests(unittest.TestCase):
    def test_dtype_only_author_transform(self):
        source='\n'.join('v'+str(i)+'=torch.float16' for i in range(4))+'\nx=other.float16\ny=torch.float32'
        tree,count=transform_builder(source,4)
        self.assertEqual(count,4)
        self.assertEqual(ast.dump(tree),ast.dump(ast.parse(source.replace('torch.float16','torch.bfloat16'))))
        with self.assertRaises(Blocked):transform_builder(source,5)
    def test_frozen_sources_expected_shapes(self):
        root=Path(__file__).parent/'.local-preparation/runs/visual-compression-r3/vendor'
        cases=[('PriorTR','image/LLaVA/llava/model/builder.py',5),('PriorTR','image/LLaVA/llava/vtr/model/builder.py',4),('AgilePruner','llava/model/builder.py',5)]
        for name,relative,count in cases:
            dirs=list((root/name).glob('*'))
            if not dirs:self.skipTest('frozen sources unavailable')
            _,actual=transform_builder((dirs[0]/relative).read_text(),count);self.assertEqual(actual,count)
    def test_resize_guard_prevents_random_embedding(self):
        class Weight:shape=(10,2)
        class Embedding:weight=Weight()
        class Model:
            def __init__(self):self.calls=[]
            def get_input_embeddings(self):return Embedding()
            def resize_token_embeddings(self,n,*args,**kwargs):self.calls.append(n);return n
        model=Model();original=guard_resize(model)
        self.assertEqual(model.resize_token_embeddings(10),10)
        with self.assertRaises(Blocked):model.resize_token_embeddings(11)
        self.assertEqual(model.calls,[10]);model.resize_token_embeddings=original
        self.assertEqual(model.resize_token_embeddings(11),11)
    def test_five_task_smoke_exact_candidate_selection(self):
        spec=read(Path(__file__).parent/'round3_spec.json')
        rows=[{'uid':f'{t}:{i}','task':t,'image_sha256':f'{t}:{i}','native_inputs':{m:{'eligible':True} for m in ('llava7','qwen8','mini8')}} for t in TASKS for i in range(20)]
        common=build(spec,rows,'sha')
        for condition in CONDITIONS:
            reference=prepare_jobs(spec,rows,{'manifest_sha256':'sha'},condition,integration_smoke=True)
            candidate=[j for j in common['jobs'] if j['model']=='llava7' and j['side']=='candidate' and j['condition']==condition]
            first={}
            for job in candidate:
                if job['repeat']==0:first.setdefault(job['task'],job['job_key'])
            candidate=[j for j in candidate if j['job_key'] in first.values()]
            self.assertEqual(len(reference),5)
            self.assertEqual([(j['uid'],j['repeat']) for j in reference],[(j['uid'],j['repeat']) for j in candidate])
            full=prepare_jobs(spec,rows,{'manifest_sha256':'sha'},condition)
            for j in reference:
                self.assertIn(j,full);self.assertEqual(j['generation']['max_new_tokens'],2048)
                self.assertEqual(j['stage'],'natural_certification')
            with self.assertRaises(Blocked):prepare_jobs(spec,rows,{'manifest_sha256':'sha'},condition,smoke=True,integration_smoke=True)

    def test_jobs_use_exact_common_identity(self):
        spec=read(Path(__file__).parent/'round3_spec.json')
        rows=[{'uid':f'{t}:{i}','task':t,'image_sha256':f'{t}:{i}','native_inputs':{m:{'eligible':True} for m in ('llava7','qwen8','mini8')}} for t in TASKS for i in range(20)]
        common=build(spec,rows,'sha')
        for condition in CONDITIONS:
            jobs=prepare_jobs(spec,rows,{'manifest_sha256':'sha'},condition)
            expected=[j for j in common['jobs'] if j['model']=='llava7' and j['side']=='reference' and j['condition']==condition]
            self.assertEqual(jobs,expected);self.assertEqual(len(jobs),300)
            self.assertEqual(prepare_jobs(spec,rows,{'manifest_sha256':'sha'},condition,True),jobs[:1])
            self.assertTrue(all(j['generation']['max_new_tokens']==2048 for j in jobs))

if __name__=='__main__':unittest.main()
