import copy
import json
import tempfile
import unittest
from pathlib import Path
from r3.common import Blocked,digest
from r3.enrich_records import enrich_record,enrich_file
from test_r3_token_partitions import Tokenizer


class EnrichRecords(unittest.TestCase):
    def make(self,model='qwen8',ids=None):
        ids=[1,101,2,3,102] if ids is None else ids
        prompt={'model':model,'uid':'q1','prompt_ids':[100,2] if model=='qwen8' else [1,2]}
        record={'job_key':'abc','model':model,'uid':'q1','record_complete':True,
                'generated_token_ids':ids,'output_tokens':len(ids),'output_text':'ORIGINAL TEXT UNCHANGED',
                'prompt_sha256':digest(json.dumps(prompt['prompt_ids'])),
                'generation':{'max_new_tokens':20},'eos_terminated':ids[-1]==102,
                'output_cap_reached':False}
        return record,prompt
    def test_qwen_isolation_and_original_preservation(self):
        record,prompt=self.make(ids=[3,101,1,102]);original=copy.deepcopy(record)
        enriched=enrich_record(record,prompt,Tokenizer(),[102])
        self.assertEqual(enriched['final_text'],'reason')
        self.assertTrue(enriched['final_region_parse_failure'])
        self.assertEqual(record,original)
        for key,value in original.items():self.assertEqual(enriched[key],value)
    def test_unclosed_think_cannot_leak_marker_to_scoring(self):
        record,prompt=self.make(ids=[3,102])
        enriched=enrich_record(record,prompt,Tokenizer(),[102])
        self.assertEqual(enriched['final_text'],'')
        self.assertTrue(enriched['final_region_parse_failure'])
        self.assertEqual(enriched['final_text_scope'],'empty_unclosed_think')
    def test_non_thinking_final_line_and_prompt_binding(self):
        record,prompt=self.make('llava7',[1,2,3,102])
        enriched=enrich_record(record,prompt,Tokenizer(),[102])
        self.assertEqual(enriched['final_text'],'FINAL ANSWER: 42')
        self.assertEqual(enriched['explicit_explanation_tokens'],2)
        prompt['prompt_ids']=[4]
        with self.assertRaises(Blocked):enrich_record(record,prompt,Tokenizer(),[102])
    def test_conflicting_existing_derived_field_refused(self):
        record,prompt=self.make();record['reasoning_tokens']=999
        with self.assertRaises(Blocked):enrich_record(record,prompt,Tokenizer(),[102])
    def test_atomic_no_replace_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=root/'raw.jsonl';prompts=root/'prompts.jsonl';out=root/'enriched.jsonl'
            record,prompt=self.make()
            records.write_text(json.dumps(record)+'\n');prompts.write_text(json.dumps(prompt)+'\n')
            receipt=enrich_file(records,prompts,out,'qwen8',Tokenizer(),[102],{'test':True})
            self.assertEqual(receipt['records'],1)
            saved=out.read_bytes()
            with self.assertRaises(Blocked):enrich_file(records,prompts,out,'qwen8',Tokenizer(),[102],{})
            self.assertEqual(out.read_bytes(),saved)
            records.write_text((json.dumps(record)+'\n')*2)
            failed=root/'failed.jsonl'
            with self.assertRaises(Blocked):enrich_file(records,prompts,failed,'qwen8',Tokenizer(),[102],{})
            self.assertFalse(failed.exists())


if __name__=='__main__':unittest.main()
