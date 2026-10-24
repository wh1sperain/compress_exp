import json
import tempfile
import unittest
from pathlib import Path
from r3.common import Blocked,digest,seed
from r3.manifest import TASKS
from r3.offline_pipeline import process_record,run_pipeline,frozen_inputs
from test_r3_token_partitions import Tokenizer


class OfflinePipeline(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);frozen=root/'frozen';frozen.mkdir()
        rows=[];labels={};jobs=[];records=[];prompt=[1,2];sha=digest(json.dumps(prompt))
        for task in TASKS:
            for i in range(20):
                uid=task+str(i)
                row={'uid':uid,'task':task,'image_sha256':digest(uid),'answer_type':'integer','precision':0,
                     'native_inputs':{'llava7':{'prompt_sha256':sha}}}
                rows.append(row);labels[uid]={'answer':'42'}
                for repeat in range(3):
                    for side in ('reference','candidate'):
                        job={'job_key':digest(uid+str(repeat)+side),'uid':uid,'model':'llava7','condition':'Full',
                             'repeat':repeat,'side':side,'stage':'natural_certification','task':task,
                             'image_sha256':digest(uid),'seed':seed(uid,repeat),'generation':{'max_new_tokens':2048}}
                        jobs.append(job)
                        records.append(dict(job,record_complete=True,generated_token_ids=[1,2,3,102],output_tokens=4,
                            output_cap_reached=False,eos_terminated=True,prompt_ids=prompt,prompt_sha256=sha,
                            generation_eos_token_id=[102]))
        manifest=frozen/'calibration_manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        label_path=frozen/'scoring_labels.json';label_path.write_text(json.dumps(labels))
        (frozen/'manifest_lock.json').write_text(json.dumps({'hashes':{p.name:digest(p.read_bytes()) for p in (manifest,label_path)}}))
        job_path=root/'jobs.json';job_path.write_text(json.dumps({'jobs':jobs}))
        records_path=root/'raw.jsonl';records_path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        return frozen,job_path,records_path,rows,labels,jobs,records

    def test_complete_six_hundred_rows_to_natural_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen,jobs,records,*_=self.fixture(directory)
            result=run_pipeline(records,frozen,'calibration',jobs,Path(directory)/'out','llava7',Tokenizer(),{'test':True})
            self.assertTrue(result['records_complete'])
            self.assertEqual(result['scored_records'],600)
            self.assertTrue(result['natural_certification']['Full']['natural_statistics_passed'])
            self.assertFalse(result['native_gate_passed'])
            with self.assertRaises(Blocked):run_pipeline(records,frozen,'calibration',jobs,Path(directory)/'out','llava7',Tokenizer(),{})

    def test_missing_record_remains_incomplete_not_zero_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen,jobs,path,_,_,_,records=self.fixture(directory)
            path.write_text(''.join(json.dumps(r)+'\n' for r in records[:-1]))
            result=run_pipeline(path,frozen,'calibration',jobs,Path(directory)/'out','llava7',Tokenizer(),{})
            self.assertFalse(result['records_complete'])
            self.assertEqual(result['scored_records'],599)
            self.assertEqual(len(result['missing_job_keys']),1)
            self.assertEqual(result['natural_certification']['Full']['status'],'INCONCLUSIVE')

    def test_label_hash_and_original_prompt_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen,_,_,rows,labels,jobs,records=self.fixture(directory)
            records[0]['prompt_ids']=[4]
            with self.assertRaises(Blocked):process_record(records[0],rows[0],labels[rows[0]['uid']],jobs[0],Tokenizer())
            (frozen/'scoring_labels.json').write_text('{}')
            with self.assertRaises(Blocked):frozen_inputs(frozen,'calibration')


if __name__=='__main__':unittest.main()
