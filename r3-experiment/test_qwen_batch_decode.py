import unittest
import torch
from r3.common import Blocked
from r3.qwen_batch_decode import pad_layer,padded_attention,Backend


class QwenBatchPrimitiveTests(unittest.TestCase):
    def test_mixed_length_sdpa_equals_individual_and_padding_ignored(self):
        torch.manual_seed(17)
        pairs=[(torch.randn(1,2,n,4,dtype=torch.float64),torch.randn(1,2,n,4,dtype=torch.float64)) for n in (2,7)]
        snapshots=[(k.clone(),v.clone()) for k,v in pairs]
        k=torch.randn(2,2,1,4,dtype=torch.float64);v=torch.randn_like(k);q=torch.randn(2,4,1,4,dtype=torch.float64)
        pk,pv,mask,updated=pad_layer(pairs,k,v)
        actual=padded_attention(q,pk,pv,mask,.5)
        for i,(keys,values) in enumerate(updated):
            expected=padded_attention(q[i:i+1],keys,values,torch.ones(1,1,1,keys.shape[2],dtype=torch.bool),.5)
            self.assertTrue(torch.allclose(actual[i:i+1],expected,atol=1e-12,rtol=1e-12))
            self.assertTrue(torch.equal(pairs[i][0],snapshots[i][0]))
            self.assertTrue(torch.equal(pairs[i][1],snapshots[i][1]))
        pk[0,:,3:]=1e9;pv[0,:,3:]=-1e9
        self.assertTrue(torch.allclose(actual,padded_attention(q,pk,pv,mask,.5),atol=1e-12,rtol=1e-12))

    def test_no_aliasing_and_per_layer_lengths_bytes_release(self):
        old=torch.zeros(1,1,2,2)
        new=torch.ones(2,1,1,2)
        _,_,_,updated=pad_layer([(old,old),(old,old)],new,new)
        updated[0][0][0,0,0,0]=9
        self.assertEqual(updated[1][0][0,0,0,0],0)
        backend=Backend.__new__(Backend)
        state={'cache':[updated[0],(torch.zeros(1,1,5,2),torch.zeros(1,1,5,2))], 'owner':id(backend)}
        self.assertEqual(backend.kv_lengths(state),[3,5])
        self.assertEqual(backend.kv_bytes(state),128)
        backend.release(state)
        self.assertEqual(backend.kv_bytes(state),0)
        self.assertTrue(state['released'])
        with self.assertRaises(Blocked):backend.release({'owner':-1})


if __name__=='__main__':unittest.main()
