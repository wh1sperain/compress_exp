import copy
import unittest
from pathlib import Path
from r3.common import read,seed,Blocked
from r3.manifest import TASKS
from r3.natural_preparation import build,CAPS

class NaturalPreparationTests(unittest.TestCase):
    def setUp(self):
        self.spec=read(Path(__file__).parent/'round3_spec.json')
        self.rows=[{'uid':f'{task}:{i}','task':task,'image_sha256':f'{task}-{i}',
                    'native_inputs':{m:{'eligible':True} for m in CAPS}} for task in TASKS for i in range(20)]
    def test_full_main_matrix_and_pairing(self):
        p=build(self.spec,self.rows,'frozen-sha')
        self.assertEqual(p['planned_records'],4800);self.assertEqual(len(p['schedule']),16)
        self.assertFalse(p['started']);self.assertFalse(p['executable'])
        index={}
        for j in p['jobs']:
            self.assertEqual(j['generation']['max_new_tokens'],CAPS[j['model']])
            self.assertEqual(j['seed'],seed(j['uid'],j['repeat']))
            key=(j['model'],j['condition'],j['uid'],j['repeat'])
            index.setdefault(key,[]).append(j)
        self.assertEqual(len(index),2400)
        for pair in index.values():
            self.assertEqual({j['side'] for j in pair},{'reference','candidate'})
            self.assertEqual(pair[0]['generation'],pair[1]['generation'])
            self.assertEqual(pair[0]['seed'],pair[1]['seed'])
            self.assertNotEqual(pair[0]['job_key'],pair[1]['job_key'])
    def test_bridge_increases_to_6000_without_replacing_data(self):
        p=build(self.spec,self.rows,'sha',True)
        self.assertEqual(len(p['jobs']),6000);self.assertEqual(len(p['schedule']),20)
        self.assertEqual(len({j['uid'] for j in p['jobs'] if j['model']=='qwen4'}),100)
    def test_missing_qwen4_audit_explicit(self):
        for row in self.rows:del row['native_inputs']['qwen4']
        p=build(self.spec,self.rows,'sha',True)
        self.assertTrue(any(b.get('model')=='qwen4' and len(b['uids'])==100 for b in p['blockers']))
    def test_no_subsample_no_label_no_cap_reduction(self):
        with self.assertRaises(Blocked):build(self.spec,self.rows[:99],'sha')
        self.rows[0]['answer']='leak'
        with self.assertRaises(Blocked):build(self.spec,self.rows,'sha')
        del self.rows[0]['answer'];s=copy.deepcopy(self.spec);s['models']['qwen8']['max_new_tokens']=128
        with self.assertRaises(Blocked):build(s,self.rows,'sha')
    def test_deterministic_jobs(self):
        a=build(self.spec,self.rows,'sha');b=build(self.spec,list(reversed(self.rows)),'sha')
        self.assertEqual(a['jobs'],b['jobs'])

if __name__=='__main__':unittest.main()
