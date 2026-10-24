import ast
import re
import unittest
from pathlib import Path
from r3.common import Blocked
from r3.scoring import edit_distance
from r3.task_scoring import score_prediction,score_record,mathvista_normalize,chart_relaxed,legacy_v2

class TaskScoringTests(unittest.TestCase):
    def test_document_required_regressions(self):
        row={'task':'DocVQA'}
        for prediction,answer,expected in [('11 a.m.','11 a.m.',1),('3 Million Users','3 million',.6),('34 Million','3 million',.9)]:
            self.assertAlmostEqual(score_prediction(prediction,row,{'answers':[answer]})['score'],expected)

    def test_document_missing_and_multiple_references(self):
        self.assertEqual(score_prediction(None,{'task':'InfographicVQA'},{'answers':['yes']})['score'],0)
        self.assertEqual(score_prediction('YES',{'task':'DocVQA'},{'answers':['no','yes']})['score'],1)
        with self.assertRaises(Blocked):score_prediction('x',{'task':'DocVQA'},{'answers':[]})

    def test_chart_author_semantics(self):
        for p,a,v in [('105','100',True),('106','100',False),('0.0','0',True),('1','0',False),('YES','yes',True),('50%','0.5',False),('$5','5',False)]:
            self.assertEqual(chart_relaxed(p,a),v)

    def test_mathvista_normalization(self):
        self.assertEqual(mathvista_normalize('(B) text',['red','blue'],'text',None),'blue')
        self.assertEqual(mathvista_normalize('red',['red','blue'],'text',None),'red')
        self.assertIsNone(mathvista_normalize(None,['red'],'text',None))
        self.assertEqual(mathvista_normalize('3.9',[],'integer',None),'3')
        self.assertEqual(mathvista_normalize('1.234',[],'float',2),'1.23')
        self.assertIsNone(mathvista_normalize('1.23 cm',[],'float',2))

    def test_official_mathvista_parity_if_snapshot_available(self):
        path=Path(__file__).parent.parent/'Compress/vendor/mathvista_calculate_score.py'
        if not path.exists():self.skipTest('reviewed source snapshot absent')
        nodes=[x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in ('get_most_similar','normalize_extracted_answer')]
        scope={'re':re,'distance':edit_distance}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
        for options,kind,precision in [(['a','b'],'text',None),([],'integer',None),([],'float',2),([],'list',None)]:
            for value in ['','A','(B)','1.25','-3','invalid','[1, 2]']:
                expected=scope['normalize_extracted_answer'](value,options,'multi_choice' if options else 'free_form',kind,precision,ignore_empty_extractions=True)
                self.assertEqual(mathvista_normalize(value,options,kind,precision),None if value=='' else expected)

    def test_gqa_unverified_blocks(self):
        with self.assertRaises(Blocked):score_prediction('yes',{'task':'GQA-balanced-val'},{'answer':'yes'})

    def test_final_region_and_shadow(self):
        row={'uid':'d1','task':'DocVQA'};record={'uid':'d1','job_key':'j','final_text':'FINAL ANSWER: 11 a.m.'}
        result=score_record(record,row,{'answers':['11 a.m.']})
        self.assertEqual(result['score'],1);self.assertEqual(result['legacy_v2'],'11 a.m')
        self.assertLess(result['legacy_v2_score'],result['score'])
        with self.assertRaises(Blocked):score_record({'uid':'d1','job_key':'j','output_text':'FINAL ANSWER: yes'},row,{'answer':'yes'})

    def test_error_not_model_wrong(self):
        result=score_record({'job_key':'j','status':'failed'},{'uid':'d1'}, {})
        self.assertIsNone(result['score'])

if __name__=='__main__':unittest.main()
