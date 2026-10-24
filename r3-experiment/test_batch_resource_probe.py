import importlib.util
import unittest
from r3.batch_resource_probe import advance_positions,reservation,repeat_real_kv
from r3.common import Blocked


class TensorShape:
    def __init__(self,n):self.shape=(1,2,n,4)
    def element_size(self):return 2


class ResourceProbeTests(unittest.TestCase):
    def test_per_layer_full_cap_padded_reservation(self):
        states=[dict(opaque=dict(cache=[(TensorShape(n),TensorShape(n))])) for n in [2,5]]
        got=reservation(states,'mini8',8)
        self.assertEqual(got['final_padded_kv_bytes'],2*2*2*14*4*2)
        self.assertEqual(got['reservation_bytes'],3*got['final_padded_kv_bytes'])
        with self.assertRaises(Blocked):reservation(states,'mini8',8,2)
    def test_position_offsets_preserve_native_baseline(self):
        state=dict(opaque=dict(next_position=100,decoded_steps=0))
        advance_positions(state,'llava7',2046)
        self.assertEqual(state['opaque']['next_position'],2146)
        self.assertEqual(state['opaque']['decoded_steps'],2046)
        with self.assertRaises(Blocked):advance_positions(state,'mini8',-1)
    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable')
    def test_repeat_actual_values_no_shared_storage(self):
        import torch
        source=torch.tensor([1.,2.,3.]).reshape(1,1,3,1)
        expanded=repeat_real_kv(source,8)
        self.assertEqual(expanded.flatten().tolist(),[1.,2.,3.,1.,2.,3.,1.,2.])
        expanded[0,0,0,0]=9
        self.assertEqual(source.flatten().tolist(),[1.,2.,3.])


if __name__=='__main__':unittest.main()
