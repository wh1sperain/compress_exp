import tempfile
import unittest
from pathlib import Path
from r3.batch_runner import run_jobs,validate_profile
from r3.common import Blocked,read


class FakeScheduler:
    fail=False
    def __init__(self,session,profile):pass
    def run(self,requests,status):
        requests=list(requests)
        for index,(row,job,path) in enumerate(requests):
            if self.fail and index==1:raise RuntimeError('injected')
            yield job,row,dict(generated_token_ids=[2],output_tokens=1,output_text='raw',
                output_cap_reached=False,eos_terminated=True,generation_eos_token_id=2)


class BatchRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)/'out'
        self.rows={str(i):dict(uid=str(i),task='DocVQA',image_sha256='image'+str(i)) for i in range(2)}
        self.jobs=[dict(job_key=str(i+1)*64,uid=str(i),task='DocVQA',image_sha256='image'+str(i),
            model='mini8',condition='MiniNative',stage='main',generation=dict(max_new_tokens=5)) for i in range(2)]
        self.profile=dict(backend='native-padded-batch',max_live_requests=4,dtype='bfloat16',kv_dtype='bfloat16')
        self.identity=dict(inference_only=True);self.loads=0
    def tearDown(self):self.temp.cleanup()
    def factory(self,*args):self.loads+=1;return object()
    def run_worker(self,scheduler=FakeScheduler):
        return run_jobs(self.jobs,self.rows,self.identity,self.out,self.profile,self.factory,scheduler,4)
    def test_immutable_success_single_load_and_skip(self):
        self.assertEqual(self.run_worker(),2);self.assertEqual(self.loads,1)
        path=self.out/self.jobs[0]['job_key']/'result.json';original=path.read_bytes()
        self.assertEqual(self.run_worker(),2);self.assertEqual(self.loads,1)
        self.assertEqual(original,path.read_bytes())
        self.assertEqual(read(path)['scoring_status'],'disabled_by_user')
    def test_failed_active_job_retained_no_auto_retry(self):
        class Failing(FakeScheduler):fail=True
        with self.assertRaises(RuntimeError):self.run_worker(Failing)
        self.assertTrue((self.out/self.jobs[0]['job_key']/'result.json').exists())
        failure=list((self.out/self.jobs[1]['job_key']).glob('failure-*.json'))
        self.assertEqual(len(failure),1);original=failure[0].read_bytes()
        with self.assertRaises(Blocked):self.run_worker()
        self.assertEqual(self.loads,1);self.assertEqual(original,failure[0].read_bytes())
    def test_input_binding_change_rejected(self):
        self.run_worker();self.rows['0']['question']='changed'
        with self.assertRaisesRegex(Blocked,'binding changed'):self.run_worker()
        self.assertEqual(self.loads,1)
    def test_mixed_conditions_and_profile_rejected(self):
        self.jobs[1]['condition']='Full'
        with self.assertRaises(Blocked):self.run_worker()
        with self.assertRaises(Blocked):validate_profile(self.profile,8)
        with self.assertRaises(Blocked):validate_profile(dict(self.profile,backend='native-b1'))


if __name__=='__main__':unittest.main()
