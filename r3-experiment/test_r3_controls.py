import ast
import json
import tempfile
import types
import unittest
from pathlib import Path
from r3.common import Blocked, digest
from r3.control_adapters import (FASTV_RELATIVE, configure_fastv_control,
                                 random_forward_ast, random_mask, uses_fastv_runtime)


class Controls(unittest.TestCase):
    def test_random_mask_repeat_stability_and_independence(self):
        first = random_mask('DocVQA:42')
        self.assertEqual(first, random_mask('DocVQA:42'))
        self.assertNotEqual(first['visual_indices'], random_mask('DocVQA:43')['visual_indices'])
        self.assertEqual(len(set(first['visual_indices'])), 144)
        self.assertTrue(all(0 <= i < 576 for i in first['visual_indices']))
        self.assertEqual(first['mask_seed'], int(digest('r3-mask|DocVQA:42')[:16], 16))
        with self.assertRaises(Blocked): random_mask('')

    def test_curve_budgets_and_guards(self):
        class Decoder:
            config = types.SimpleNamespace(use_cache=True)
            def fastv_forward(self, input_ids=None, inputs_embeds=None, use_cache=None, fastv_config=None):
                return fastv_config
        model = types.SimpleNamespace(model=Decoder())
        for name, keep in [('FastV75',432), ('FastV37p5',216), ('FastV50',288), ('FastV25',144)]:
            receipt = configure_fastv_control(model, name, 39)
            cfg = model.model.forward(use_cache=True)
            self.assertEqual(round(576*(1-cfg['fastv_r'])), keep)
            self.assertEqual(cfg['fastv_k'], 2)
            self.assertEqual(cfg['image_token_start_index'], 39)
            self.assertEqual(receipt['target_visual_tokens'], keep)
            with self.assertRaises(Blocked):model.model.forward(use_cache=False)
            with self.assertRaises(Blocked):model.model.forward(fastv_config={})
            with self.assertRaises(Blocked):model.model.forward(input_ids=types.SimpleNamespace(shape=(2,4)))
        with self.assertRaises(Blocked):configure_fastv_control(model, 'FastV10', 39)
        self.assertTrue(uses_fastv_runtime('Random25_L2'))
        self.assertFalse(uses_fastv_runtime('PriorTR25'))

    def test_real_author_ast_only_changes_selector(self):
        folder = Path(__file__).parent / '.local-preparation/runs/visual-compression-r3/vendor/FastV/d1659729b5bf1be225e99ee15783deeea80f63b1'
        path = folder / FASTV_RELATIVE
        if not path.exists(): self.skipTest('Frozen author checkout absent')
        source = path.read_text(encoding='utf-8')
        module = random_forward_ast(source)
        original = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name=='LlamaModel')
        method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name=='fastv_forward')
        original_sites = [n for n in ast.walk(method) if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='top_attention_rank_index' for t in n.targets)]
        new_sites = [n for n in ast.walk(module) if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='top_attention_rank_index' for t in n.targets)]
        self.assertEqual(len(new_sites),2)
        for before,after in zip(original_sites,new_sites):after.value = before.value
        self.assertEqual(ast.dump(method),ast.dump(module.body[-1]))
        with self.assertRaises(Blocked):random_forward_ast(source.replace('.topk(round(','.topk(int('))


if __name__ == '__main__': unittest.main()
