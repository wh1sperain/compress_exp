import tempfile
import unittest
from pathlib import Path
from r3.common import Blocked,read
from r3.manifest import TASKS
from r3.natural_preparation import build
from r3.qwen_natural_reference import jobs_for,run_jobs


class NaturalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.spec=read(Path(__file__).parent/'round3_spec.json')
        self.rows=[dict(uid=t+str(i),task=t,image_sha256=t+str(i),native_inputs={}) for t in TASKS for i in range(20)]

    def test_jobs_match_registered_natural_plan_and_cap(self):
        expected=build(self.spec,self.rows,'manifest',True)['jobs']
        for model in ('qwen8','qwen4'):
            jobs=jobs_for(self.spec,self.rows,'manifest',model,'PriorTR25')
            self.assertEqual(jobs,[j for j in expected if j['model']==model and j['condition']=='PriorTR25' and j['side']=='reference'])
            self.assertEqual(len(jobs),300)
        smoke=jobs_for(self.spec,self.rows,'manifest','qwen8','Full',self.rows[0]['uid'],2)
        self.assertEqual(len(smoke),1)
        self.assertEqual(smoke[0]['generation']['max_new_tokens'],32768)
        with self.assertRaises(Blocked):jobs_for(self.spec,self.rows,'manifest','qwen8','Full',self.rows[0]['uid'])

    def test_success_immutable_and_reuse_does_not_generate(self):
        job=jobs_for(self.spec,self.rows,'manifest','qwen8','Full',self.rows[0]['uid'],0)[0]
        class Session:
            def generate(self,*args):return dict(generated_token_ids=[9],output_tokens=1,eos_terminated=True,output_cap_reached=False)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);rows={r['uid']:r for r in self.rows}
            run_jobs([job],rows,{'code':'a'},out,Session)
            raw=(out/job['job_key']/'result.json').read_bytes()
            def forbidden():raise AssertionError('Successful job regenerated')
            self.assertEqual(run_jobs([job],rows,{'code':'a'},out,forbidden),1)
            self.assertEqual(raw,(out/job['job_key']/'result.json').read_bytes())
            with self.assertRaises(Blocked):run_jobs([job],rows,{'code':'b'},out,forbidden)

    def test_failure_stops_and_cannot_be_implicitly_retried(self):
        job=jobs_for(self.spec,self.rows,'manifest','qwen8','Full',self.rows[0]['uid'],0)[0]
        class Broken:
            def generate(self,*args):raise RuntimeError('preserve original failure')
        with tempfile.TemporaryDirectory() as temp:
            rows={r['uid']:r for r in self.rows}
            with self.assertRaises(RuntimeError):run_jobs([job],rows,{},temp,Broken)
            with self.assertRaises(Blocked):run_jobs([job],rows,{},temp,Broken)
            self.assertEqual(len(list((Path(temp)/job['job_key']).glob('failure-*.json'))),1)


if __name__=='__main__':unittest.main()
