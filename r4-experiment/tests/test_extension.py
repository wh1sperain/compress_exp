import json
import tempfile
import unittest
from pathlib import Path
from extension.common import *
from extension.prepare_data import freeze,validate
from extension.plan_jobs import matrix
from extension.parse_outputs import partition,repeated_tail
from extension.validate_backends import greedy_decision,determinism,admit
from extension.statistical_analysis import ctir,wilson,bootstrap,macro_bootstrap,bh
from extension.aggregate_metrics import summarize,aggregate
from extension.evaluate_answers import extract,relaxed,evaluate
from extension.run_inference import trace_fields
from r4.adapters import vtr_parameters

def candidates(n=180):
    return [dict(sample_id=f'{d}:{SPLITS[d]}:{i}',dataset=d,split=SPLITS[d],question='q',choices=['x','y'] if d=='AI2D' else [],gold_answer='A' if d=='AI2D' else 'SECRET',
                 question_type='free_form' if d=='MathVista' else 'type'+str(i%3),answer_type='integer',precision=0,task_type='task'+str(i%5),image_paths=['unused'],image_count=1,
                 image_hash=digest([d,i]),source_id=str(i),subset='Human' if i%2==0 else 'Augmented',
                 discipline='discipline'+str(i%6),variant='Vision Dominant') for d in DATASETS for i in range(n)]

class ExtensionTests(unittest.TestCase):
    def setUp(self):self.spec=read(ROOT/'extension_protocol.json');self.dec={m:dict(mode='greedy') for m in MODELS}
    def test_main_46800_shared_baseline(self):
        jobs=matrix(candidates(150),self.spec,'formal',self.dec)
        self.assertEqual(len(jobs),46800);self.assertEqual(sum(j['method']=='Full' for j in jobs),1800)
        self.assertEqual(len({j['run_id'] for j in jobs}),46800)
        self.assertTrue(all(j['method']=='Full' for j in jobs[:1800]))
    def test_pilot_3840(self):self.assertEqual(len(matrix(candidates(20),self.spec,'pilot')),3840)
    def test_precheck_240(self):self.assertEqual(len(matrix(candidates(20),self.spec,'precheck')),240)
    def test_robustness_3600_corrected_arithmetic(self):self.assertEqual(len(matrix(candidates(150),self.spec,'robustness')),3600)
    def test_determinism_900(self):
        jobs=matrix(candidates(20),self.spec,'determinism');self.assertEqual(len(jobs),900)
        self.assertEqual({j['seed'] for j in jobs},{42});self.assertEqual({j['repeat'] for j in jobs},{0,1,2})
    def test_unapproved_fallback_blocked(self):
        self.dec['qwen4']['mode']='sampling'
        with self.assertRaises(ValueError):matrix(candidates(150),self.spec,'formal',self.dec)
    def test_bound_fallback_extra9000(self):
        self.dec['qwen4']['mode']='sampling'
        jobs=matrix(candidates(150),self.spec,'formal',self.dec,dict(decisions_sha256=digest(self.dec),core_seeds=[42,43,44]))
        self.assertEqual(len(jobs),55800)
    def test_freeze_strata_independence_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b';rr=candidates()
            locks={d:dict(source='official',revision='fixed') for d in DATASETS}
            receipt=freeze(rr,a,locks,False);freeze(list(reversed(rr)),b,locks,False)
            formal=rows(a/'formal_inputs.jsonl');pilot=rows(a/'pilot_inputs.jsonl')
            self.assertEqual(len(formal),900);self.assertEqual(len(pilot),120)
            self.assertFalse({r['image_hash'] for r in formal}&{r['image_hash'] for r in pilot})
            self.assertNotIn('SECRET',(a/'formal_inputs.jsonl').read_text())
            self.assertEqual((a/'formal_inputs.jsonl').read_bytes(),(b/'formal_inputs.jsonl').read_bytes())
            self.assertEqual(receipt['distributions']['ChartQA']['strata'],dict(Human=75,Augmented=75))
            self.assertEqual(set(receipt['distributions']['MMMU']['strata'].values()),{25})
    def test_reject_multimage_and_wrong_mathverse(self):
        r=candidates(1)[0];r['image_count']=2;self.assertFalse(validate(r,False))
        r=next(r for r in candidates(1) if r['dataset']=='MathVerse');r['variant']='Text Only';self.assertFalse(validate(r,False))
    def test_no_old_methods_or_repeats(self):
        jobs=matrix(candidates(20),self.spec,'pilot')
        self.assertFalse({'Random','Resolution'} & {j['method'] for j in jobs});self.assertEqual({j['repeat'] for j in jobs},{0})
    def test_native_token_partition(self):
        x=partition([10,11,2,12,9],[1],1,2,[9]);self.assertEqual((x['reasoning_tokens'],x['answer_tokens'],x['total_output_tokens']),(2,1,5))
    def test_generated_open(self):self.assertEqual(partition([1,10,2,12,9],[],1,2,[9])['reasoning_tokens'],1)
    def test_unclosed_never_final_answer(self):
        x=partition([10,11],[1],1,2,[9]);self.assertEqual(x['reasoning_tokens'],2);self.assertFalse(x['final_answer_present'])
    def test_missing_open_is_unknown_length(self):self.assertIsNone(partition([10,2,12],[],1,2,[9])['reasoning_tokens'])
    def test_double_close_ambiguous(self):self.assertEqual(partition([10,2,12,2],[1],1,2,[9])['think_parse_status'],'ambiguous_boundary')
    def test_loop_nonoverlap(self):
        self.assertTrue(repeated_tail(list(range(80))*3)['flag']);self.assertFalse(repeated_tail([1]*239)['flag'])
    def test_loop_interleaved(self):self.assertTrue(repeated_tail(list(range(80))+[1000]+list(range(80))+[1001]+list(range(80)))['flag'])
    def test_ratio_cost_not_mean_ratio(self):self.assertAlmostEqual(ctir([1,99],[100,100]),100)
    def test_zero_wilson(self):self.assertGreater(wilson(0,10)[1],0);self.assertIsNone(wilson(0,0))
    def test_paired_identical_ci(self):self.assertEqual(bootstrap([10,20],[10,20],n=100)['ctir_ci95'],[0,0])
    def test_macro_requires6(self):
        with self.assertRaises(ValueError):macro_bootstrap([([1],[2])],n=10)
    def test_bh(self):self.assertEqual(bh([.01,.04,.03]),[.03,.04,.04])
    def test_short_denominator_kept_in_ctir(self):
        def r(n):return dict(reasoning_tokens=n,total_output_tokens=n+2,correct=True,finish_reason='eos',think_parse_status='ok',keep_ratio_actual=.25,visual_tokens_removed=75)
        x=summarize([(r(1),r(100)),(r(100),r(200))],50)
        self.assertEqual(x['ratio_denominator'],1);self.assertEqual(x['excluded_short_baseline'],1)
        self.assertAlmostEqual(x['ctir'],199/101*100);self.assertEqual(x['growth']['2']['rate'],1)
    def test_default_method_config_not_old_override(self):
        cfg=vtr_parameters('SparseVLM',.75,self.spec,100)
        self.assertNotIn('prune_layer',cfg);self.assertNotIn('keep_tokens',cfg);self.assertNotIn('token_merge',cfg)
    def test_hilo_budget_fusion_distinct(self):
        t=dict(full_visual_tokens=10,layer_count=2,selections=[dict(layer=0,original_indices=[0,2,3],fusion_indices=[1,4])],
               prefill_layers={'0':{'hidden_shape':[1,15,10]},'1':{'hidden_shape':[1,8,10]}},prompt_ids=list(range(15)),
               image_grid_thw=[[1,4,10]],method_parameters={'hilo_use_fusion':True})
        x=trace_fields(t,'HiLo');self.assertEqual(x['visual_tokens_retained'],3);self.assertEqual(x['merged_token_count'],0);self.assertEqual(x['fusion_source_token_count'],2)
    def test_full_selector_forbidden(self):
        t=dict(full_visual_tokens=1,layer_count=1,selections=[dict(original_indices=[0])],prefill_layers={'0':{}},prompt_ids=[0])
        with self.assertRaises(ValueError):trace_fields(t,'Full')
    def test_extraction_ambiguity(self):self.assertIsNone(extract('It could be A or B',['x','y'])[0])
    def test_chartqa_percent(self):self.assertTrue(relaxed('50%','0.5'));self.assertFalse(relaxed('nan','1'))
    def test_missing_gate_blocks(self):
        with self.assertRaises(ValueError):admit({}, {},'qwen4','HiLo25')
    def test_precheck_threshold_strict(self):
        panel=candidates(20);records=[]
        for model in MODELS:
            for i,r in enumerate(panel):records.append(dict(sample_id=r['sample_id'],model=model,method='Full',decoding_mode='greedy',phase='precheck',run_status='success',correct=True,repetition_warning={'flag':False},total_output_tokens=32768 if i<6 else 10))
        d=greedy_decision(records,panel,{});self.assertEqual(d['qwen4']['mode'],'greedy')
        records[6]['total_output_tokens']=32768;self.assertEqual(greedy_decision(records,panel,{})['qwen4']['mode'],'sampling')
    def test_fixed_seed_disagreement_fails(self):
        rr=[dict(model='qwen4',sample_id='a',method='FastV',keep_ratio_target=.25,repeat=i,selected_visual_token_indices=[1],visual_tokens_retained=1,raw_generated_token_ids=[i],final_answer_text='x',reasoning_tokens=1) for i in range(3)]
        self.assertEqual(determinism(rr)[0]['status'],'FAIL')
    def test_unknown_quality_does_not_become_wrong(self):
        x=dict(reasoning_tokens=40,total_output_tokens=42,correct=None,finish_reason='eos',think_parse_status='ok',keep_ratio_actual=.5,visual_tokens_removed=50)
        y=dict(x,reasoning_tokens=80)
        r=summarize([(x,y)],50);self.assertEqual(r['ctir'],100);self.assertIsNone(r['accuracy']);self.assertEqual(r['unresolved_scores'],1)
    def test_cap_retained_warning(self):
        x=dict(reasoning_tokens=100,total_output_tokens=102,correct=True,finish_reason='eos',think_parse_status='ok',keep_ratio_actual=1,visual_tokens_removed=0)
        y=dict(x,reasoning_tokens=32768,total_output_tokens=32768,finish_reason='length',think_parse_status='missing_think_end',correct=False)
        r=summarize([(x,y)],50);self.assertTrue(r['censoring_warning']);self.assertGreater(r['ctir'],32000)
    def test_incomplete_macro_not_pooled(self):
        self.assertIsNone(ctir([0],[20]))
    def test_boundary_noanswer_count_wrong(self):
        row=dict(sample_id='a',image_hash='h',choices=[],question='q',dataset='MathVerse')
        result=dict(row,run_id='r',raw_sha256='s',run_status='success',think_parse_status='missing_think_end',final_answer_text='')
        scored,q=evaluate([result],[row],[dict(sample_id='a',image_hash='h',gold_answer='42')])
        self.assertIs(scored[0]['correct'],False);self.assertFalse(q)
    def test_failure_not_wrong(self):
        row=dict(sample_id='a',image_hash='h',choices=[],question='q',dataset='MathVerse')
        result=dict(row,run_id='r',raw_sha256='s',run_status='failure',think_parse_status='missing_think_end',final_answer_text='')
        scored,q=evaluate([result],[row],[dict(sample_id='a',image_hash='h',gold_answer='42')])
        self.assertIsNone(scored[0]['correct']);self.assertEqual(len(q),1)
    def test_wholly_missing_configuration_blocks(self):
        with self.assertRaises(ValueError):aggregate([],[],expected_jobs=[{'run_id':'missing'}])

if __name__=='__main__':unittest.main()
