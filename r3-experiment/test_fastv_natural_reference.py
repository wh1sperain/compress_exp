import unittest
from r3.common import Blocked
from r3.fastv_natural_reference import remap_placeholder,exhaustive_mapping


class FastVReferenceTests(unittest.TestCase):
    def test_dynamic_offset_and_collision(self):
        self.assertEqual(remap_placeholder([1,4,-200,5]),([1,4,0,5],2))
        self.assertEqual(remap_placeholder([1,-200,5]),([1,0,5],1))
        for ids in ([0,-200],[-200,-200],[1,2],[-201,-200]):
            with self.assertRaises(Blocked):remap_placeholder(ids)

    def test_mapping_complete_and_unknown_rejected(self):
        source=['model.embed_tokens.weight','model.mm_projector.0.weight']
        target=['language_model.model.embed_tokens.weight','multi_modal_projector.linear_1.weight']
        self.assertEqual(set(exhaustive_mapping(source,target)),set(target))
        with self.assertRaises(Blocked):exhaustive_mapping(source,target[:1])
        with self.assertRaises(Blocked):exhaustive_mapping(['unregistered.weight'],[])
        with self.assertRaises(Blocked):exhaustive_mapping(source+source,target)


if __name__=='__main__':unittest.main()
