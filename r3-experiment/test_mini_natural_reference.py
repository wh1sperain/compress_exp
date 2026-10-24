import tempfile
import unittest
from pathlib import Path
from r3.common import Blocked,read,seed
from r3.manifest import TASKS
from r3.mini_natural_reference import prepare_jobs,run_jobs

class MiniNaturalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.spec=read(Path(__file__).parent/'round3_spec.json')
        self.rows=[{'uid':f'{t}:{i}','task':t,'image_sha256':f'{t}-{i}','native_inputs':{'mini8':{'eligible':True}}} for t in TASKS for i in range(20)]
    def test_full_matrix_and_smoke_unchanged_cap(self):
        jobs=prepare_jobs(self.spec,self.rows,{'manifest_sha256':'fixed'})
        self.assertEqual(len(jobs),300)
        for j in jobs:self.assertEqual(j['seed'],seed(j['uid'],j['repeat']));self.assertEqual(j['generation']['max_new_tokens'],2048)
        smoke=prepare_jobs(self.spec,self.rows,{'manifest_sha256':'fixed'},True)
        self.assertEqual(len(smoke),1);self.assertEqual(smoke[0]['generation']['max_new_tokens'],2048)
        self.assertEqual(smoke[0]['stage'],'natural_certification')
        self.assertEqual(smoke[0],jobs[0])
        from r3.natural_preparation import build
        canonical_plan=build(self.spec,self.rows,'fixed')
        expected=[j for j in canonical_plan['jobs'] if j['model']=='mini8' and j['side']=='reference']
        self.assertEqual(jobs,expected)
    def test_persistent_single_loader_and_immutable_success(self):
        jobs=prepare_jobs(self.spec,self.rows,{'manifest_sha256':'fixed'})[:2];loads=[]
        class Session:
            def generate(self,row,job,path):
                return dict(generated_token_ids=[1,2],output_tokens=2,output_text='x',eos_terminated=True,output_cap_reached=False)
        def factory():loads.append(1);return Session()
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);self.assertEqual(run_jobs(jobs,self.rows,out,factory),2);self.assertEqual(len(loads),1)
            raw=(out/jobs[0]['job_key']/'result.json').read_bytes()
            self.assertEqual(run_jobs(jobs,self.rows,out,factory),2);self.assertEqual(len(loads),1)
            self.assertEqual(raw,(out/jobs[0]['job_key']/'result.json').read_bytes())
    def test_short_non_eos_is_failure_not_retry(self):
        jobs=prepare_jobs(self.spec,self.rows,{'manifest_sha256':'fixed'},True)
        class Bad:
            def generate(self,row,job,path):return dict(generated_token_ids=[1],output_tokens=1,eos_terminated=False,output_cap_reached=False)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Blocked):run_jobs(jobs,self.rows,d,Bad)
            self.assertEqual(len(list(Path(d).rglob('failure-*.json'))),1)
            with self.assertRaisesRegex(Blocked,'Prior failure'):run_jobs(jobs,self.rows,d,Bad)
            self.assertEqual(len(list(Path(d).rglob('failure-*.json'))),1)
    def test_refuses_labels(self):
        self.rows[0]['answer']='secret'
        with self.assertRaises(Blocked):prepare_jobs(self.spec,self.rows,{'manifest_sha256':'fixed'})

if __name__=='__main__':unittest.main()
