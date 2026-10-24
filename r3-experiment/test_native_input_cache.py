import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from PIL import Image
from r3.common import digest,Blocked
from r3.native_session import NativeSession,SessionInputFingerprints,ProcessorFingerprint


class FakeIds:
    def __init__(self,ids):self.ids=ids
    def __getitem__(self,key):return self
    def __eq__(self,key):return FakeIds([x==key for x in self.ids])
    def __len__(self):return len(self.ids)
    def sum(self):return sum(self.ids)
    def tolist(self):return list(self.ids)
    def to(self,device):return self


class InputCacheTests(unittest.TestCase):
    def test_tokenizer_and_processor_serialize_once(self):
        for qwen in (True,False):
            tokenizer=Mock();tokenizer.backend_tokenizer.to_str.return_value='tokenizer'
            tokenizer.get_vocab.return_value={'a':1}
            processor=Mock();processor.to_dict.return_value={'size':336}
            cache=SessionInputFingerprints(tokenizer,processor,qwen,'template')
            expected=digest(json.dumps({'size':336},sort_keys=True))
            for _ in range(5):self.assertEqual(cache.processor.match(expected),(expected,expected))
            processor.to_dict.assert_called_once()
            if qwen:tokenizer.backend_tokenizer.to_str.assert_called_once()
            else:tokenizer.get_vocab.assert_called_once()

    def test_fastv_metadata_compatibility_and_wrong_transform(self):
        known=['images','do_resize','size','resample','do_center_crop','crop_size',
            'do_rescale','rescale_factor','do_normalize','image_mean','image_std',
            'do_convert_rgb','return_tensors','data_format','input_data_format']
        config={'size':336,'_valid_processor_keys':known}
        cache=ProcessorFingerprint(config)
        raw=digest(json.dumps(config,sort_keys=True));expected=digest(json.dumps({'size':336},sort_keys=True))
        self.assertEqual(cache.match(raw),(raw,raw))
        self.assertEqual(cache.match(expected),(raw,expected))
        with self.assertRaises(Blocked):cache.match(digest(json.dumps({'size':224},sort_keys=True)))

    def test_prepare_still_checks_each_audit_and_decodes_hashed_bytes(self):
        tokenizer=Mock();tokenizer.backend_tokenizer.to_str.return_value='tokenizer'
        processor=Mock();processor.image_processor.to_dict.return_value={'size':336}
        processor.chat_template='template';processor.apply_chat_template.return_value='prompt'
        processor.return_value={'input_ids':FakeIds([1,99,2])}
        session=NativeSession.__new__(NativeSession)
        session.key='qwen8';session.condition='Full';session.spec={'prompt':{'suffix':' suffix'}}
        session.processor=processor;session.tokenizer=tokenizer
        session.model=types.SimpleNamespace(device='cpu',config=types.SimpleNamespace(image_token_id=99))
        session.input_fingerprints=SessionInputFingerprints(tokenizer,processor.image_processor,True,'template')
        data=io.BytesIO();Image.new('RGB',(2,2),(12,34,56)).save(data,format='PNG');raw=data.getvalue()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'im.png';path.write_bytes(raw)
            audit=dict(eligible=True,prompt_sha256=digest(json.dumps([1,99,2])),
                processor_sha256=digest(json.dumps({'size':336},sort_keys=True)),
                tokenizer_sha256=digest('tokenizer'),chat_template_sha256=digest('template'),
                visual_tokens=1,expanded_prompt_tokens=3)
            row=dict(uid='q',question='Q',image_path=str(path),image_sha256=digest(raw),native_inputs={'qwen8':audit})
            actual_open=Image.open;decoded=[]
            def checking_open(source):
                self.assertIsInstance(source,io.BytesIO);decoded.append(source.getvalue())
                return actual_open(source)
            with patch.dict('sys.modules',{'torch':types.SimpleNamespace()}),patch('PIL.Image.open',side_effect=checking_open):
                _,a=session.prepare(row);_,b=session.prepare(row)
                self.assertEqual(a,b);self.assertEqual(decoded,[raw,raw])
                audit['tokenizer_sha256']='changed'
                with self.assertRaises(Blocked):session.prepare(row)
                audit['tokenizer_sha256']=digest('tokenizer');path.write_bytes(b'changed')
                with self.assertRaises(Blocked):session.prepare(row)
            tokenizer.backend_tokenizer.to_str.assert_called_once()
            processor.image_processor.to_dict.assert_called_once()


if __name__=='__main__':unittest.main()
