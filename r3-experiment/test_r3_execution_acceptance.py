import json
import tempfile
import unittest
from pathlib import Path
from r3.common import Blocked,digest,read
from r3.manifest import TASKS
from r3.execution_acceptance import create,MODE
from r3.register_run import register
from r3.runner import load_plan,run_jobs


class ExecutionAcceptance(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);(root/'manifests').mkdir()
        for name in ('source_lock.json','model_weights_lock.json'):(root/'manifests'/name).write_text('{}')
        rows=[{'uid':t+str(i),'task':t,'image_sha256':digest(t+str(i)),'native_inputs':{}} for t in TASKS for i in range(100)]
        manifest=root/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        profile=root/'profile.json';profile.write_text('{"backend":"native-b1"}')
        audit=root/'audit.jsonl'
        audit.write_text(''.join(json.dumps({'uid':r['uid'],'model':'qwen4','audit':{'eligible':True,'expanded_prompt_tokens':10,
            'processor_sha256':'p','tokenizer_sha256':'t','chat_template_sha256':'c','prompt_sha256':'x'}})+'\n' for r in rows))
        status=root/'audit.status.json';status.write_text(json.dumps({'stage':'complete','exit_code':0,'completed':500,'total':500,'audit_sha256':digest(audit.read_bytes())}))
        evidence=root/'diagnostic.json';evidence.write_text('{"status":"PASS","native_gate_passed":false}')
        spec=Path(__file__).parent/'round3_spec.json';acceptance=root/'acceptance.json'
        create(root,spec,manifest,profile,audit,status,[evidence],'TEST explicit inference-only reduced acceptance',
               ['Full native/engine certification is not complete'],acceptance)
        return root,spec,manifest,profile,audit,status,evidence,acceptance

    def test_explicit_policy_registration_load_and_inference_only_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root,spec,manifest,profile,audit,status,evidence,acceptance=self.fixture(directory)
            receipt=register(root,spec,manifest,profile,['main','controls'],None,None,root/'registration',audit,status,acceptance)
            self.assertFalse(receipt['gates_verified'])
            self.assertEqual(receipt['inventory']['planned_records'],21900)
            self.assertNotIn('passed',read(acceptance))
            plan,rows,identity=load_plan(receipt['plans'][0]['path'],root)
            self.assertEqual(identity['execution_mode'],MODE)
            job=plan['jobs'][0]
            class FakeSession:
                def generate(self,row,job,checkpoint):
                    return {'generated_token_ids':[1],'output_tokens':1,'eos_terminated':True,'output_cap_reached':False}
            run_jobs([job],rows,identity,root/'worker',lambda *args:FakeSession())
            result=read(root/'worker'/job['job_key']/'result.json')
            self.assertEqual(result['scoring_status'],'disabled_by_user')
            self.assertFalse(result['full_native_engine_certification'])
            self.assertNotIn('score',result)
            evidence.write_text('{"status":"CHANGED"}')
            with self.assertRaises(Blocked):load_plan(receipt['plans'][0]['path'],root)

    def test_strict_path_still_requires_gates_and_scope_cannot_shrink(self):
        with tempfile.TemporaryDirectory() as directory:
            root,spec,manifest,profile,audit,status,evidence,acceptance=self.fixture(directory)
            with self.assertRaises(Blocked):register(root,spec,manifest,profile,['main','controls'],None,None,root/'strict',audit,status)
            with self.assertRaises(Blocked):register(root,spec,manifest,profile,['main'],None,None,root/'partial',audit,status,acceptance)
            value=read(acceptance);value['passed']=True;acceptance.write_text(json.dumps(value))
            with self.assertRaises(Blocked):register(root,spec,manifest,profile,['main','controls'],None,None,root/'fake',audit,status,acceptance)

    def batch_profile(self,root,profile):
        from r3.execution_acceptance import BATCH_FILES
        code_root=Path(__file__).resolve().parent
        hashes={name:digest((code_root/name).read_bytes()) for name in BATCH_FILES}
        probes=[]
        for model in ('llava7','qwen8','mini8'):
            path=root/(model+'-mock-probe.json')
            path.write_text(json.dumps(dict(status='PASS',model=model,backend_impl_sha256=hashes,test_fixture=True)))
            probes.append(dict(path=str(path),sha256=digest(path.read_bytes())))
        value=dict(backend='native-padded-batch',memory_admission='full_remaining_cap_with_scratch',kv_scratch_factor=3,
                   max_live_requests=4,dtype='bfloat16',kv_dtype='bfloat16',backend_impl_sha256=hashes,batch_probe_receipts=probes)
        profile.write_text(json.dumps(value));return value

    def test_batch_registration_and_receipt_tamper(self):
        from r3.execution_acceptance import MODE_BATCH
        with tempfile.TemporaryDirectory() as directory:
            root,spec,manifest,profile,audit,status,evidence,old=self.fixture(directory)
            value=self.batch_profile(root,profile);acceptance=root/'batch-acceptance.json'
            create(root,spec,manifest,profile,audit,status,[evidence],'TEST batch authorization',['Full certification incomplete'],acceptance)
            receipt=register(root,spec,manifest,profile,['main','controls'],None,None,root/'batch-plans',audit,status,acceptance)
            self.assertEqual(receipt['execution_mode'],MODE_BATCH);self.assertFalse(receipt['gates_verified'])
            plan,rows,identity=load_plan(receipt['plans'][0]['path'],root)
            self.assertEqual(plan['backend'],'native-padded-batch');self.assertEqual(identity['execution_mode'],MODE_BATCH)
            Path(value['batch_probe_receipts'][0]['path']).write_text('{"status":"FAIL"}')
            with self.assertRaises(Blocked):load_plan(receipt['plans'][0]['path'],root)

    def test_batch_profile_hash_budget_and_failed_probe_rejected(self):
        from r3.execution_acceptance import validate_batch_profile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);profile=root/'profile.json';value=self.batch_profile(root,profile)
            validate_batch_profile(value)
            value['kv_scratch_factor']=2
            with self.assertRaises(Blocked):validate_batch_profile(value)
            value['kv_scratch_factor']=3;value['backend_impl_sha256']['r3/batch_scheduler.py']='stale'
            with self.assertRaises(Blocked):validate_batch_profile(value)
            value=self.batch_profile(root,profile)
            item=value['batch_probe_receipts'][0];path=Path(item['path']);probe=read(path);probe['status']='FAIL'
            path.write_text(json.dumps(probe));item['sha256']=digest(path.read_bytes())
            with self.assertRaises(Blocked):validate_batch_profile(value)

    def test_probe_source_snapshot_binding(self):
        from r3.execution_acceptance import validate_batch_profile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);value=self.batch_profile(root,root/'profile.json')
            release=root/'old-release';(release/'r3').mkdir(parents=True)
            decoder='r3/llava_batch_decode.py'
            (release/decoder).write_bytes((Path(__file__).resolve().parent/decoder).read_bytes())
            source=release/'r3'
            sha=digest(json.dumps({p.relative_to(source).as_posix():digest(p.read_bytes()) for p in sorted(source.rglob('*.py'))},sort_keys=True))
            item=value['batch_probe_receipts'][0];path=Path(item['path'])
            path.write_text(json.dumps(dict(status='PASS',model='llava7',code_sha256=sha,test_fixture=True)))
            item.update(sha256=digest(path.read_bytes()),source_release=str(release),source_code_sha256=sha)
            validate_batch_profile(value)
            (release/decoder).write_text('changed')
            with self.assertRaises(Blocked):validate_batch_profile(value)


if __name__=='__main__':unittest.main()
