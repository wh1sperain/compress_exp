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


if __name__=='__main__':unittest.main()
