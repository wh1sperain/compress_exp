import ast
import importlib.util
import unittest
from pathlib import Path
from r3.common import Blocked
from r3.mini_independent_probe import bf16_builder_tree, require_strict, finalize_bf16

class MiniIndependentTests(unittest.TestCase):
    def test_exact_dtype_only_transform(self):
        source='\n'.join('x'+str(i)+'=torch.float16' for i in range(5))+'\ny=torch.float32\nz=other.float16\n'
        tree,count=bf16_builder_tree(source)
        self.assertEqual(count,5)
        expected=source.replace('torch.float16','torch.bfloat16')
        self.assertEqual(ast.dump(tree),ast.dump(ast.parse(expected)))

    def test_drift_rejected(self):
        with self.assertRaises(Blocked): bf16_builder_tree('x=torch.float16')

    def test_loading_must_be_exact(self):
        require_strict({'missing_keys':[], 'unexpected_keys':[]})
        for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
            with self.assertRaises(Blocked):require_strict({key:['vision.weight']})

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch CPU runtime unavailable')
    def test_full_model_dtype_conversion_preserves_values(self):
        import torch
        model=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.LayerNorm(2))
        expected={k:v.detach().to(torch.bfloat16).clone() for k,v in model.state_dict().items()}
        receipt=finalize_bf16(model,'cpu')
        self.assertTrue(receipt['non_bf16_parameters_before_conversion'])
        self.assertTrue(receipt['all_floating_parameters_bf16'])
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,expected[k]))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch CPU runtime unavailable')
    def test_meta_parameter_is_never_silently_materialized(self):
        import torch
        model=torch.nn.Linear(3,2,device='meta')
        with self.assertRaisesRegex(Blocked,'Unmaterialized'):finalize_bf16(model,'cpu')
        self.assertTrue(model.weight.is_meta)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch CPU runtime unavailable')
    def test_meta_buffer_is_rejected(self):
        import torch
        model=torch.nn.Linear(3,2)
        model.register_buffer('bad',torch.empty(1,device='meta'))
        with self.assertRaisesRegex(Blocked,'Unmaterialized'):finalize_bf16(model,'cpu')
        self.assertEqual(model.weight.dtype,torch.float32)

    def test_frozen_builder_when_available(self):
        path=Path(__file__).parent/'.local-preparation/runs/visual-compression-r3/vendor/LLaVA-Mini/47da1137f7b503be685ab5faa9f64a88b102e786/llavamini/model/builder.py'
        if not path.exists():self.skipTest('vendored author source not local')
        _,count=bf16_builder_tree(path.read_text(encoding='utf-8'))
        self.assertEqual(count,5)

if __name__=='__main__':unittest.main()
