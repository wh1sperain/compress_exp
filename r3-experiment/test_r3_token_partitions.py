import unittest
from r3.common import Blocked
from r3.token_partitions import partition_generated, explanation_boundary, prompt_has_open_think


class Tokenizer:
    all_special_ids = [100,101,102]
    pieces = {1:'reason',2:'\n',3:'FINAL ANSWER: 42',4:'more',5:'reason\nFINAL',6:' ANSWER: 42',
              100:'<think>',101:'</think>',102:'<eos>'}
    def encode(self,text,add_special_tokens=False):
        return {'<think>':[100],'</think>':[101]}[text]
    def decode(self,ids,skip_special_tokens=False,clean_up_tokenization_spaces=False):
        return ''.join(self.pieces[t] for t in ids if not(skip_special_tokens and t in self.all_special_ids))


class TokenPartitions(unittest.TestCase):
    def setUp(self): self.t = Tokenizer()
    def q(self,ids,cap=20):return partition_generated(ids,self.t,'qwen8',cap,[102],prompt_think_open=True)
    def test_qwen_three_partition_and_prompt_excluded(self):
        self.assertTrue(prompt_has_open_think([100,2],self.t))
        r=self.q([1,101,2,3,102])
        self.assertEqual([r[k] for k in ('reasoning_tokens','answer_tokens','structural_tokens')],[1,2,2])
        self.assertFalse(r['boundary_anomaly'])
        self.assertEqual(r['final_text'],'\nFINAL ANSWER: 42')
        self.assertEqual(r['censoring'],'natural_think_end')
    def test_cap_unclosed_closed_and_eos_at_cap(self):
        r=self.q([1,4],2)
        self.assertTrue(r['reasoning_right_censored'])
        self.assertEqual(r['censoring'],'length_before_think_end')
        r=self.q([1,101,3],3)
        self.assertTrue(r['output_right_censored'])
        self.assertFalse(r['reasoning_right_censored'])
        r=self.q([1,101,102],3)
        self.assertTrue(r['output_cap_reached'])
        self.assertFalse(r['output_right_censored'])
        self.assertEqual(r['finish_reason'],'eos')
    def test_boundary_anomalies_and_early_stop(self):
        self.assertIn('eos_without_think_end',self.q([1,102])['boundary_anomalies'])
        self.assertIn('multiple_think_close',self.q([1,101,101,102])['boundary_anomalies'])
        self.assertIn('unexpected_early_termination',self.q([1])['boundary_anomalies'])
        with self.assertRaises(Blocked):partition_generated([1],self.t,'qwen8',20,[102])
    def test_explicit_marker_maps_original_ids_without_encode(self):
        r=partition_generated([1,2,3,102],self.t,'llava7',20,[102])
        self.assertEqual(r['explicit_explanation_tokens'],2)
        self.assertEqual(r['explicit_explanation_token_bounds'],[2,2])
        self.assertIsNone(r['reasoning_tokens'])
        self.assertFalse(r['marker_followed_by_nonempty_text'])
        r=explanation_boundary([5,6,102],self.t)
        self.assertIsNone(r['explicit_explanation_tokens'])
        self.assertEqual(r['explicit_explanation_token_bounds'],[0,1])
        self.assertTrue(r['boundary_crosses_token'])
    def test_last_marker_and_missing_marker(self):
        r=explanation_boundary([3,2,4,2,3,2,4],self.t)
        self.assertEqual(r['marker_count'],2)
        self.assertEqual(r['explicit_explanation_tokens'],4)
        self.assertTrue(r['marker_followed_by_nonempty_text'])
        self.assertIsNone(explanation_boundary([1,102],self.t)['explicit_explanation_tokens'])
    def test_multitoken_tags_do_not_steal_unrelated_component_ids(self):
        class Multi(Tokenizer):
            all_special_ids=[102]
            pieces=dict(Tokenizer.pieces,**{})
            pieces.update({7:'</',8:'think>',9:'<',10:'think>'})
            def encode(self,text,add_special_tokens=False):return {'<think>':[9,10],'</think>':[7,8]}[text]
        r=partition_generated([7,1,7,8,3,102],Multi(),'qwen8',20,[102],prompt_think_open=True)
        self.assertEqual((r['reasoning_tokens'],r['answer_tokens'],r['structural_tokens']),(2,1,3))
    def test_original_length_validation(self):
        with self.assertRaises(Blocked):self.q([1,2],1)
        with self.assertRaises(Blocked):self.q([])
        with self.assertRaises(Blocked):self.q([True])


if __name__=='__main__':unittest.main()
