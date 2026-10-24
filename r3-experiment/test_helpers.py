import copy
import json
from pathlib import Path
import unittest
from validate_plan import validate, kv_bytes_per_token

class PlanTests(unittest.TestCase):
    def setUp(self):
        self.s=json.loads(Path(__file__).with_name('round3_spec.json').read_text())
    def test_original_plan(self):
        r=validate(self.s)
        self.assertTrue(r['plan_arithmetic_passed'])
        self.assertEqual(r['counts']['complete'],23160)
        self.assertFalse(r['production_ready'])
    def test_extra_cls_is_detected(self):
        s=copy.deepcopy(self.s)
        s['conditions']['VisionZip25']['dominant']+=1
        self.assertFalse(validate(s)['plan_arithmetic_passed'])
    def test_wrong_count_is_detected(self):
        s=copy.deepcopy(self.s)
        s['expected_records']['complete']-=1
        self.assertFalse(validate(s)['plan_arithmetic_passed'])
    def test_overlong_llava_is_detected(self):
        s=copy.deepcopy(self.s)
        s['models']['llava7']['max_new_tokens']=32768
        self.assertFalse(validate(s)['plan_arithmetic_passed'])
    def test_qwen_adaptation_rejected(self):
        s=copy.deepcopy(self.s)
        s['main_tracks'][1]['conditions'][2]='AgilePruner25'
        self.assertFalse(validate(s)['plan_arithmetic_passed'])
    def test_mini_false_full_rejected(self):
        s=copy.deepcopy(self.s)
        s['models']['mini8']['allow_full_counterfactual']=True
        self.assertFalse(validate(s)['plan_arithmetic_passed'])
    def test_kv_formula(self):
        c={'num_hidden_layers':32,'num_attention_heads':32,'num_key_value_heads':8,'hidden_size':4096}
        self.assertEqual(kv_bytes_per_token(c),131072)
    def test_no_quantization(self):
        s=copy.deepcopy(self.s);s['generation']['kv_quantization']=True
        self.assertFalse(validate(s)['plan_arithmetic_passed'])

if __name__=='__main__':
    unittest.main()
