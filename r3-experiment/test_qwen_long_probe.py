import unittest
import torch
from r3.qwen_long_probe import probe_plan,filtered_summary


class LongProbeTests(unittest.TestCase):
    def test_requested_windows_only(self):
        total,steps=probe_plan(True)
        self.assertEqual(total,32784)
        self.assertEqual(steps,{0,1}|{i for s in (1024,4096,8192,16384,32768) for i in range(s,s+16)})
        self.assertEqual(probe_plan(False)[0],1025)

    def test_filter_support_stable_ties_and_excluded_eos(self):
        result=filtered_summary(torch.zeros(30),{'think':0,'eos':29})
        self.assertEqual(result['top20_ids'],list(range(20)))
        self.assertAlmostEqual(sum(result['filtered_probabilities']),1.)
        self.assertTrue(result['specials']['think']['top20_eligible'])
        self.assertFalse(result['specials']['eos']['top20_eligible'])
        self.assertEqual(result['specials']['eos']['filtered_probability'],0.)
        self.assertGreater(result['specials']['eos']['raw_probability'],0.)


if __name__=='__main__':unittest.main()
