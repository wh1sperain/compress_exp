import unittest
from r3.common import Blocked, seed
from r3.manifest import TASKS
from r3.natural_certification import evaluate


class NaturalCertificationTests(unittest.TestCase):
    def fixture(self):
        manifest=[];rows=[]
        for task in TASKS:
            for i in range(20):
                uid=f'{task}-{i}'
                m=dict(uid=uid,task=task,image_sha256=uid);manifest.append(m)
                for repeat in range(3):
                    for side in ('reference','candidate'):
                        rows.append(dict(m,repeat=repeat,side=side,seed=seed(uid,repeat),model='llava7',condition='Full',
                            record_complete=True,output_tokens=100,output_cap_reached=False,score=1.,prompt_sha256=uid))
        return manifest,rows

    def test_identical_complete_pairs_pass_statistics_not_gate(self):
        m,r=self.fixture();out=evaluate(r,m,'llava7','Full')
        self.assertTrue(out['natural_statistics_passed']);self.assertFalse(out['native_gate_passed'])

    def test_missing_duplicate_and_wrong_seed_rejected(self):
        m,r=self.fixture()
        for bad in (r[:-1], r+[r[0]], [dict(r[0],seed=0)]+r[1:]):
            with self.assertRaises(Blocked):evaluate(bad,m,'llava7','Full')

    def test_task_score_cancellation_cannot_pass(self):
        m,r=self.fixture()
        for row in r:
            row['score']=.5
            if row['side']=='candidate':
                row['score'] += .1 if row['task']==TASKS[0] else -.1 if row['task']==TASKS[1] else 0
        out=evaluate(r,m,'llava7','Full')
        self.assertFalse(out['natural_statistics_passed'])
        self.assertFalse(out['checks']['task_score/'+TASKS[0]])


if __name__=='__main__':unittest.main()
