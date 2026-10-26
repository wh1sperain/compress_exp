import json
import tempfile
import unittest
from pathlib import Path
from r4.core import *
from r4.prepare import prepare
from r4.interventions import *
from r4.annotations import validate,mechanism_panel
from r4.gates import admit,validate_trace
from r4.adapters import vtr_parameters
from r4.analyze import paired

def rows(n=60):
    return [dict(uid=f'{t}:{i}',task=t,question='q',options=[],image_path='x',image_sha256=sha([t,i]),answer='secret')
            for t in TASKS for i in range(n)]

class ProtocolTests(unittest.TestCase):
    def setUp(self):self.spec=read(ROOT/'protocol.json')
    def test_discovery_count(self):
        p=select(rows(),30,'a');self.assertEqual(len(matrix(p,self.spec)),6750)
    def test_pair_seeds(self):
        jobs=matrix(rows(1),self.spec)
        for u in {j['uid'] for j in jobs}:
            for r in range(3):self.assertEqual(len({j['seed'] for j in jobs if j['uid']==u and j['repeat']==r}),1)
    def test_order_invariant(self):self.assertEqual(select(rows(),30,'a'),select(list(reversed(rows())),30,'a'))
    def test_exclude_images(self):
        r=rows();bad={x['image_sha256'] for x in r[::7]}
        self.assertFalse(bad & {x['image_sha256'] for x in select(r,30,'b',bad)})
    def test_shortage_blocks(self):
        with self.assertRaises(ValueError):select(rows(2),30,'a')
    def test_duplicate_blocks(self):
        r=rows(1)
        with self.assertRaises(ValueError):select(r+[r[0]],1,'a')
    def test_label_isolation(self):self.assertNotIn('answer',input_only(rows(1)[0]))
    def test_confirm_requires_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s';p.write_text('\n'.join(json.dumps(r) for r in rows()))
            with self.assertRaises(ValueError):prepare(p,Path(d)/'out',self.spec,confirm=True)
    def test_confirm_count(self):
        p=select(rows(),50,'c');self.assertEqual(len(matrix(p,self.spec,'R4-2',('Full','FastV50','FastV25','PriorTR50','PriorTR25','HiLo50','HiLo25'))),5250)
    def test_rescue_same_removal(self):
        r=rescue_pair([0,1,2,3],[3,4],10,'x')
        self.assertEqual(len(r['rescue']),4);self.assertEqual(len(r['placebo']),4)
        self.assertIn(4,r['rescue']);self.assertNotIn(4,r['placebo'])
        self.assertEqual(set(r['original'])-set(r['rescue']),set(r['original'])-set(r['placebo']))
    def test_rescue_infeasible(self):
        with self.assertRaises(ValueError):rescue_pair([0],[1,2],4,'x')
    def test_roi_no_guessed_grid(self):
        with self.assertRaises(ValueError):map_region([0,0,1,1],[[0,0,1,1]],'image',{})
    def test_roi_verified(self):
        b=[[0,0,.5,1],[.5,0,1,1]]
        self.assertEqual(map_region([.1,.1,.4,.9],b,'i',{'image_sha256':'i','processor_mapping_verified':True,'token_boxes_sha256':sha(b)}),[0])
    def test_prefix_no_future_leak(self):
        p=prefix_triplet([1,2,3,4,5],1,3,[8],[9]);self.assertEqual(p['corrected']['forced_prefix'],[1,8])
        self.assertEqual(p['corrected']['rng_offset'],3)
    def test_stop_reuse(self):self.assertTrue(stop_prefix([1,151668,2],256)['reuse_natural'])
    def test_stop_total_budget(self):
        p=stop_prefix([4]*1024,256);self.assertEqual(len(p['forced_prefix'])+p['max_new_tokens'],32768)
    def test_stop_incomplete(self):
        with self.assertRaises(ValueError):stop_prefix([4]*3,256)
    def test_stop_excludes_think_open(self):
        p=stop_prefix([151667]+[4]*1000,256)
        self.assertEqual(len(p['forced_prefix']),258);self.assertEqual(p['rng_offset'],257)
    def test_fake_human_blocks(self):
        with self.assertRaises(ValueError):validate({'reviewers':['a'],'adjudicated':True},10)
    def test_error_requires_fact(self):
        with self.assertRaises(ValueError):validate({'reviewers':['a','b'],'adjudicated':True,'visual_error':True},10)
    def test_unknown_not_false(self):
        self.assertIsNone(validate({'reviewers':['a','b'],'adjudicated':True,'visual_error':None},10))
    def test_empty_gate_blocks(self):
        with self.assertRaises(ValueError):admit({}, {'code_sha256':code_hash()},'Full')
    def test_sparse_absolute_budget(self):
        p=vtr_parameters('SparseVLM',.25,self.spec,100)
        self.assertEqual(p['keep_tokens'][-1],25);self.assertFalse(p['token_merge'])
    def test_fastv_auto_query(self):self.assertEqual(vtr_parameters('FastV',.25,self.spec)['query_aggregation'],'auto')
    def test_missing_trace_blocks(self):
        with self.assertRaises(ValueError):validate_trace({},100,'PriorTR')
    def test_m120_shortages_not_invented(self):
        r=mechanism_panel([]);self.assertEqual(r['actual_n'],0);self.assertEqual(len(r['shortages']),20)
    def paired_rows(self):
        return [dict(uid=str(i),task=TASKS[0],repeat=r,seed=r,image_sha256=str(i),condition=c,
                     reasoning_tokens=10+(c!='Full'),quality=None)
                for i in range(4) for r in range(3) for c in ('Full','PriorTR25')]
    def test_cluster_constant_difference(self):
        r=paired(self.paired_rows(),'PriorTR25',TASKS[0],'reasoning_tokens',draws=100)
        self.assertEqual(r['mean_difference'],1);self.assertEqual(r['ci95'],[1,1])
    def test_missing_quality_not_zero(self):
        r=paired(self.paired_rows(),'PriorTR25',TASKS[0],'quality',draws=100)
        self.assertEqual(r['status'],'INCOMPLETE')
    def test_mismatched_seed_rejected(self):
        r=self.paired_rows();r[0]['seed']=999
        with self.assertRaises(ValueError):paired(r,'PriorTR25',TASKS[0],'reasoning_tokens',draws=100)

if __name__=='__main__':unittest.main()
