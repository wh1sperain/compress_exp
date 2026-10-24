import json
import tempfile
import unittest
from pathlib import Path
from r3.tokenizer_partition_probe import synthetic_checks,replay_natural
from test_r3_token_partitions import Tokenizer


class ProbeTokenizer(Tokenizer):
    def encode(self,text,add_special_tokens=False):
        return {'<think>':[100],'</think>':[101],'Reasoning.\n':[1,2],
                'FINAL ANSWER: 42':[3],'Reasoning.\nFINAL ANSWER: 42':[1,2,3]}[text]


class TokenizerProbe(unittest.TestCase):
    def test_synthetic_scope_and_eos(self):
        for model in ('llava7','mini8','qwen8','qwen4'):
            result=synthetic_checks(ProbeTokenizer(),model,[102])
            self.assertEqual(result['status'],'PASS')
            self.assertTrue(result['synthetic_tokenizer_test'])
    def test_replay_uses_actual_ids_not_reconstructed_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            identity={'uid':'u','model_revision':'revision','natural_cap':20,'prompt_ids':[[100,2]]}
            natural={'generated_token_ids':[1,101,3,102],'output_tokens':4,'output_cap_reached':False,'eos_terminated':True}
            (path/'identity.json').write_text(json.dumps(identity));(path/'natural.json').write_text(json.dumps(natural))
            result=replay_natural(path,ProbeTokenizer(),'qwen8','revision',[102])
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(result['partition']['reasoning_tokens'],1)
            identity.pop('prompt_ids');(path/'identity.json').write_text(json.dumps(identity))
            self.assertEqual(replay_natural(path,ProbeTokenizer(),'qwen8','revision',[102])['status'],'INCONCLUSIVE')


if __name__=='__main__':unittest.main()
