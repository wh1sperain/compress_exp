import ast
import re
import unittest
from pathlib import Path
from r3.common import Blocked,digest
from r3.scoring import edit_distance
from r3.task_scoring import (score_prediction,score_record,mathvista_normalize,chart_relaxed,legacy_v2,
                            gqa_exact,GQA_EVALUATOR_REVISION,GQA_EVALUATOR_PATH,GQA_EVALUATOR_SHA256)

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

    def test_gqa_exact_without_vqa_or_submission_normalization(self):
        row={'task':'GQA-balanced-val'}
        for pred,gold,expected in [('yes','yes',1),('YES','yes',0),('yes.','yes',0),
                                   ('two','2',0),('the cat','cat',0),(' yes ','yes',0),
                                   ('gray','grey',0),(None,'yes',0)]:
            result=score_prediction(pred,row,{'answer':gold})
            self.assertEqual(result['score'],expected)
            self.assertEqual(result['normalized_prediction'],pred)
            self.assertEqual(result['evaluator_sha256'],GQA_EVALUATOR_SHA256)
        with self.assertRaises(Blocked):score_prediction('yes',row,{'answers':['yes','no']})

    def test_gqa_frozen_evaluator_expression_parity(self):
        path=(Path(__file__).parent/'.local-preparation/runs/visual-compression-r3/vendor/PriorTR'
              /GQA_EVALUATOR_REVISION/GQA_EVALUATOR_PATH)
        if not path.exists():self.skipTest('frozen source snapshot absent')
        self.assertEqual(digest(path.read_bytes()),GQA_EVALUATOR_SHA256)
        tree=ast.parse(path.read_text(encoding='utf-8'))
        nodes=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='correct' for t in node.targets)]
        self.assertEqual(len(nodes),1)
        expression=compile(ast.Expression(body=nodes[0].value),str(path),'eval')
        for predicted in ['yes','YES','yes.',' yes ','2','two','the cat','cat']:
            for gold in ['yes','2','cat']:
                self.assertEqual(gqa_exact(predicted,gold),eval(expression,{'predicted':predicted,'gold':gold}))

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
