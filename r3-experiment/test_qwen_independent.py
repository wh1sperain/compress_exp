"""CPU guard tests; no claim of GPU/model validation."""
import tempfile
import unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None
from r3.common import write, Blocked
from r3.qwen_independent_probe import PROBES, tensor_record, prior_formula, check_identity, compare


class QwenIndependentTests(unittest.TestCase):
    def identity(self, side):
        return dict(side=side, method='Full', model_key='qwen8', model_revision='model',
                    author_revision='source', uid='calibration', image_sha256='image',
                    inputs={}, forced_sha256='prefix', steps=1025)

    def test_independent_entrypoint_required(self):
        with self.assertRaises(Blocked): check_identity(self.identity('local'), self.identity('local'))
        a, b = self.identity('author'), self.identity('local')
        check_identity(a, b)
        b['image_sha256'] = 'other'
        with self.assertRaises(Blocked): check_identity(a, b)

    @unittest.skipIf(torch is None, 'PyTorch is not installed in this interpreter')
    def test_tensor_fingerprint_preserves_bf16_and_positions(self):
        a = torch.tensor([1., 2.], dtype=torch.bfloat16)
        self.assertEqual(tensor_record(a), tensor_record(a.clone()))
        self.assertNotEqual(tensor_record(a), tensor_record(a.flip(0)))
        self.assertIsNone(tensor_record(None))

    @unittest.skipIf(torch is None, 'PyTorch is not installed in this interpreter')
    def test_prior_uniform_has_zero_information(self):
        attention = torch.full((1, 2, 8, 8), .125)
        p, q, score = prior_formula(attention, (1, 4))
        self.assertTrue(torch.equal(p, q))
        self.assertTrue(torch.equal(score, torch.zeros_like(score)))

    @unittest.skipIf(torch is None, 'PyTorch is not installed in this interpreter')
    def test_compare_rejects_trace_mismatch_and_missing_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            author, local = Path(tmp)/'author', Path(tmp)/'local'
            for folder, side in ((author, 'author'), (local, 'local')):
                write(folder/'identity.json', self.identity(side))
                write(folder/'status.json', {'status':'PASS'})
                write(folder/'trace.json', {'position':[1,2]})
                torch.save({i:torch.tensor([1.,2.,3.]) for i in PROBES}, folder/'logits.pt')
            result = compare(author, local)
            self.assertEqual(result['status'], 'PASS')
            self.assertFalse(result['native_gate_passed'])
            write(local/'trace.json', {'position':[1,3]})
            self.assertEqual(compare(author, local)['status'], 'FAIL')
            torch.save({0:torch.ones(3)}, local/'logits.pt')
            with self.assertRaises(Blocked): compare(author, local)


if __name__ == '__main__': unittest.main()
