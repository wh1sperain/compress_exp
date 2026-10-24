import unittest
from r3.common import Blocked
from r3.llava_batch_decode import layer_layout,legacy_cache


class LlamaBatchLayout(unittest.TestCase):
    def test_per_layer_masks_do_not_equalize_fastv_lengths(self):
        early=layer_layout([675,720]);late=layer_layout([243,288])
        self.assertEqual(early['valid_lengths'],[676,721])
        self.assertEqual(late['valid_lengths'],[244,289])
        self.assertEqual(sum(early['attention_mask'][0]),676)
        self.assertEqual(sum(late['attention_mask'][0]),244)
        self.assertFalse(late['attention_mask'][0][-1])
        self.assertTrue(late['attention_mask'][1][-1])
    def test_invalid_layouts_fail_closed(self):
        for values in ([],[0],[True],[2,-1]):
            with self.assertRaises(Blocked):layer_layout(values)
        with self.assertRaises(Blocked):legacy_cache(None)
        with self.assertRaises(Blocked):legacy_cache([])


if __name__=='__main__':unittest.main()
