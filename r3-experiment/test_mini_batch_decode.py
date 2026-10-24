import importlib.util
import unittest
from r3.common import Blocked
from r3.mini_batch_decode import Backend, layout, pack


class MiniBatchTests(unittest.TestCase):
    def test_left_padding_and_real_positions(self):
        result = layout([2, 4, 1])
        self.assertEqual(result['position_ids'], [[2], [4], [1]])
        self.assertEqual(result['attention_mask'], [[0,0,1,1,1], [1,1,1,1,1], [0,0,0,1,1]])

    def test_invalid_lengths_rejected(self):
        for lengths in ([], [0], [-1,3], [True], [1.5]):
            with self.assertRaises(Blocked): layout(lengths)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch CPU runtime unavailable')
    def test_mixed_cache_pack_and_slot_isolation(self):
        import torch
        states = [dict(opaque=dict(cache=((torch.full((1,2,n,4),float(i+1)),
                 torch.full((1,2,n,4),float(i+11))),),next_position=n)) for i,n in enumerate([2,4])]
        cache,mask,pos,lengths = pack(states)
        self.assertEqual(cache[0][0].shape, (2,2,4,4))
        self.assertEqual(pos.tolist(), [[2],[4]])
        self.assertTrue((cache[0][0][0,:,:2,:] == 0).all())
        self.assertTrue((cache[0][0][0,:,2:,:] == 1).all())
        self.assertTrue((cache[0][0][1] == 2).all())
        with self.assertRaises(Blocked): pack([states[0],states[0]])
        states[0]['opaque']['next_position'] = 99
        with self.assertRaises(Blocked): pack(states)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'), 'torch/transformers unavailable')
    def test_native_llama3_rope_mixed_batch_matches_single(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(9)
        config = LlamaConfig(vocab_size=31, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=8192, rope_theta=500000.,
            rope_scaling={'factor':8., 'low_freq_factor':1., 'high_freq_factor':4.,
                          'original_max_position_embeddings':8192,'rope_type':'llama3'})
        config._attn_implementation = 'eager'
        model = LlamaForCausalLM(config).eval()
        backend = object.__new__(Backend)
        backend.model = model
        backend.native_forward = LlamaForCausalLM.forward
        states = []
        with torch.inference_mode():
            for n in (3,7):
                out = model(input_ids=torch.arange(1,n+1)[None,:], use_cache=True)
                cache = out.past_key_values
                if hasattr(cache,'to_legacy_cache'):cache=cache.to_legacy_cache()
                states.append(dict(opaque=dict(cache=cache,next_position=n)))
            tokens = torch.tensor([11,12])
            expected = torch.cat([model(input_ids=tokens[i:i+1,None],
                past_key_values=s['opaque']['cache'],position_ids=torch.tensor([[s['opaque']['next_position']]]),
                use_cache=True).logits[:,-1,:] for i,s in enumerate(states)])
            actual = backend.decode(states,tokens)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        self.assertEqual(backend.kv_lengths(states[0]),[4,4])
        self.assertEqual(backend.kv_lengths(states[1]),[8,8])
        self.assertNotEqual(states[0]['opaque']['cache'][0][0].untyped_storage().data_ptr(),
                            states[1]['opaque']['cache'][0][0].untyped_storage().data_ptr())


if __name__ == '__main__': unittest.main()
