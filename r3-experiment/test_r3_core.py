import unittest
import numpy as np
from r3.scoring import extract,nls,r10
from r3.sampling import cpu_sample,uniforms
from r3.common import seed
from r3.build_jobs import build
from r3.manifest import TASKS,select,split_labels
from r3.rescue import masks
from r3.native_sources import visionzip_arguments
from r3.analyze import paired,intervals
from r3.common import Blocked
from r3.artifacts import verify_cached,sha256
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
from r3.compat import llava_forward_with_cache_position
from pathlib import Path
import json

class CoreTests(unittest.TestCase):
    def test_range_transfer_assembles_verified_archive(self):
        from r3.range_download import transfer
        from hashlib import sha256 as hash_bytes
        data=b'fixture archive bytes'
        class Response:
            status_code=206
            def __init__(self,start,end):
                self.headers={'Content-Range':f'bytes {start}-{end}/{len(data)}','ETag':'fixed'}
                self.content=data[start:end+1]
            def __enter__(self):return self
            def __exit__(self,*args):pass
        def get(url,headers,**kwargs):
            start,end=map(int,headers['Range'].split('=')[1].split('-'))
            return Response(start,end)
        with tempfile.TemporaryDirectory() as d,patch('requests.get',get),patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
            target=Path(d)/'archive.zip';receipt=transfer('https://example.invalid/data',target,lambda **kw:None)
            self.assertEqual(target.read_bytes(),data)
            self.assertEqual(receipt['sha256'],hash_bytes(data).hexdigest())
    def test_ranges_reject_wrong_bytes_or_changed_source(self):
        from r3.range_download import validate_range
        headers={'Content-Range':'bytes 8-15/32','ETag':'stable'}
        validate_range(headers,8,15,32,('ETag','stable'))
        with self.assertRaises(Blocked):validate_range(headers,0,7,32,('ETag','stable'))
        with self.assertRaises(Blocked):validate_range(headers,8,15,32,('ETag','changed'))
    def test_input_prompt_excludes_labels_and_preserves_options(self):
        from r3.audit_qwen_inputs import question_text
        text=question_text({'question':'Which?','options':['red','blue'],'answer':'SECRET'},'\nSUFFIX')
        self.assertIn('(A) red\n(B) blue',text)
        self.assertNotIn('SECRET',text)
        self.assertTrue(text.endswith('\nSUFFIX'))
    def test_official_git_blob_integrity(self):
        from r3.chartqa_data import verify_blob
        verify_blob(b'hello\n','ce013625030ba8dba906f756967f9e9ca394464a')
        with self.assertRaises(Blocked):verify_blob(b'changed','ce013625030ba8dba906f756967f9e9ca394464a')
    def test_manifest_freeze_is_complete_label_isolated_and_immutable(self):
        from r3.manifest import freeze
        rows=[]
        for task in TASKS:
            for i in range(120):
                audit=dict(eligible=True,processor_sha256='p',prompt_sha256='t',expanded_prompt_tokens=700,
                           visual_tokens=576,processed_size=[336,336],native_context=4096,hidden_answer='SECRET')
                rows.append(dict(uid=f'{task}:{i}',task=task,image_sha256=f'{task}:{i}',image_path='/image',
                    question='question',answers=['SECRET'],rationale='SECRET',source_revision='fixed',
                    single_image=True,rgb_decodable=True,public_label=True,
                    native_inputs={m:dict(audit) for m in ['llava7','qwen8','mini8']}))
                rows[-1]['native_inputs']['mini8'].update(visual_tokens=1,prefusion_lengths={str(j):1277 for j in range(4)},model_revision='fixed-mini')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'frozen';freeze(rows,out)
            self.assertEqual(len((out/'main_manifest.jsonl').read_text().splitlines()),500)
            self.assertEqual(len((out/'calibration_manifest.jsonl').read_text().splitlines()),100)
            self.assertNotIn('SECRET',(out/'main_manifest.jsonl').read_text())
            self.assertIn('SECRET',(out/'scoring_labels.json').read_text())
            mini=json.loads((out/'main_manifest.jsonl').read_text().splitlines()[0])['native_inputs']['mini8']
            self.assertEqual(mini['prefusion_lengths'],{str(j):1277 for j in range(4)})
            with self.assertRaises(Blocked):freeze(rows,out)
            with self.assertRaises(Blocked):select(rows+[rows[0]])
            rows[0]['native_inputs']['mini8']['prefusion_lengths']['0']=5000
            with self.assertRaises(Blocked):freeze(rows,Path(d)/'bad-prefusion')
    def test_audit_join_rejects_partial_duplicates_and_corruption(self):
        from r3.freeze_audited_inputs import assemble
        from r3.common import write,digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'candidates.jsonl';source.write_text(json.dumps({'uid':'u'})+'\n')
            paths=[]
            for model in ['llava7','qwen8','mini8']:
                path=root/(model+'.jsonl');path.write_text(json.dumps({'uid':'u','model':model,'audit':{'eligible':True}})+'\n')
                write(path.with_suffix('.status.json'),dict(stage='complete',exit_code=0,total=1,completed=1,audit_sha256=digest(path.read_bytes())))
                paths.append(path)
            rows,receipt=assemble([source],paths);self.assertEqual(receipt['audit_count'],3)
            with self.assertRaises(Blocked):assemble([source],paths[:2])
            with self.assertRaises(Blocked):assemble([source],paths+[paths[0]])
            write(paths[0].with_suffix('.status.json'),dict(stage='audit',exit_code=0))
            with self.assertRaises(Blocked):assemble([source],paths)
            paths[1].write_text('changed')
            with self.assertRaises(Blocked):assemble([source],paths[1:])
    def test_hf5_loading_receipt_preserves_failures(self):
        from r3.qwen_smoke import loading_receipt
        info={'missing_keys':{'b','a'},'unexpected_keys':set(),'mismatched_keys':{'x':((2,),(3,))}}
        got=json.loads(json.dumps(loading_receipt(info)))
        self.assertEqual(got['missing_keys'],['a','b'])
        self.assertEqual(got['unexpected_keys'],[])
        self.assertEqual(got['mismatched_keys'],{'x':[[2],[3]]})
        self.assertTrue(got['missing_keys'])
    def test_cache_position_bridge_preserves_arguments(self):
        calls=[];cache=object();position=object();embedding=object()
        class FakeLlama:
            @staticmethod
            def forward(self,**kw):calls.append(kw);return kw
        class FakeModel:
            def prepare_inputs_labels_for_multimodal(self,*args):
                self.prepared=args
                return None,position,'mask',cache,embedding,None
        model=FakeModel();cursor=object()
        with patch.dict('sys.modules',{'transformers':SimpleNamespace(LlamaForCausalLM=FakeLlama)}):
            result=llava_forward_with_cache_position(model,input_ids='ids',images='pixels',cache_position=cursor)
            self.assertIs(result['cache_position'],cursor)
            self.assertIs(result['past_key_values'],cache)
            self.assertIs(result['position_ids'],position)
            self.assertIs(result['inputs_embeds'],embedding)
            self.assertEqual(model.prepared[0],'ids')
            result=llava_forward_with_cache_position(model,inputs_embeds=embedding,cache_position=cursor)
            self.assertIs(result['inputs_embeds'],embedding)
    def test_cache_receipt_detects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.safetensors';p.write_bytes(b'valid-test-fixture')
            entry={'path':d,'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)}}}
            self.assertTrue(verify_cached(entry))
            p.write_bytes(b'changed-test-bytes')
            with self.assertRaises(Blocked):verify_cached(entry)
            with self.assertRaises(Blocked):verify_cached({'path':d,'files':{}})
    def test_repeat_pairing_precedes_bootstrap(self):
        rows=[]
        for repeat,delta in enumerate([0,300,600]):
            for condition,length in [('Full',100),('PriorTR25',100+delta)]:
                rows.append(dict(model='qwen8',stage='main',uid='u',task='DocVQA',image_sha256='i',
                                 repeat=repeat,condition=condition,seed=repeat,prompt_sha256='p',
                                 reasoning_tokens=length,output_cap_reached=False,score=1))
        value=paired(rows,'qwen8','PriorTR25','reasoning_tokens')[('DocVQA','i')]
        self.assertEqual(value[0],300)
        self.assertAlmostEqual(value[1],2/3)
        with self.assertRaises(Blocked):paired(rows[:-1],'qwen8','PriorTR25','reasoning_tokens')
        with self.assertRaises(Blocked):paired(rows+[rows[0]],'qwen8','PriorTR25','reasoning_tokens')
    def test_replay_cannot_destroy_longer_checkpoint(self):
        from r3.prefix_checkpoint import publish
        from r3.common import write,read
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'prefix.json';saved=list(range(300))
            old=dict(job_key='j',seed=1234,generated_token_ids=saved,generated_steps=300)
            write(p,old);raw=p.read_bytes()
            def receipt(ids):return dict(old,generated_token_ids=ids,generated_steps=len(ids))
            for n in (128,256,299,300):
                self.assertFalse(publish(p,receipt(saved[:n]),saved))
                self.assertEqual(p.read_bytes(),raw)
            with self.assertRaises(Blocked):publish(p,receipt([999]+saved[1:]),saved)
            self.assertEqual(p.read_bytes(),raw)
            extended=list(range(384));self.assertTrue(publish(p,receipt(extended),saved))
            self.assertEqual(read(p)['generated_steps'],384)
            self.assertEqual(next(p.with_name(p.name+'.history').glob('*.json')).read_bytes(),raw)
            with self.assertRaises(Blocked):publish(p,receipt(saved),saved)
            self.assertEqual(read(p)['generated_steps'],384)
    def test_agile_rejects_environment_override(self):
        from r3.native_sources import require_adaptive_agile
        with patch.dict('os.environ',{},clear=True):require_adaptive_agile()
        for value in ('','0','0.25'):
            with patch.dict('os.environ',{'DIST_THRESHOLD':value},clear=True):
                with self.assertRaises(Blocked):require_adaptive_agile()
    def test_observer_handles_direct_forward_and_preserves_output(self):
        from r3.real_selector_probe import observe_forward
        output=object();seen=[]
        class Tower:
            def forward(self,images):return output
            def __call__(self,images):return self.forward(images)
        tower=Tower();tower.forward=observe_forward(tower.forward,lambda args,result:seen.append((args,result)))
        self.assertIs(tower.forward('direct'),output)
        self.assertIs(tower('module'),output)
        self.assertEqual([x[0] for x in seen],[('direct',),('module',)])
        self.assertTrue(all(x[1] is output for x in seen))
    def test_visionzip_public_budget(self):
        for k,d,c in [(288,243,45),(144,122,22)]:
            self.assertEqual(visionzip_arguments(k),dict(dominant=d,contextual=c))
            self.assertEqual(d+c,k)
        with self.assertRaises(ValueError):visionzip_arguments(576)
    def test_required_scores(self):
        for a,b,x in [('11 a.m.','11 a.m.',1),('3 Million Users','3 million',.6),('34 Million','3 million',.9)]:
            p,_=extract('FINAL ANSWER: '+a);self.assertAlmostEqual(nls(p,[b]),x)
    def test_last_marker_and_units(self):
        self.assertEqual(extract('FINAL ANSWER: 2\nFINAL ANSWER: $32%.')[0],'$32%.')
        self.assertIsNone(extract('I guess 42')[0])
        self.assertIsNone(extract(r'\boxed{42')[0])
        self.assertEqual(extract(r'\boxed{a{b}}')[0],'a{b}')
    def test_ties_and_stream(self):
        self.assertEqual(cpu_sample([0,0,0],0,top_k=2),0)
        self.assertEqual(cpu_sample([0,0,0],.75,top_k=2),1)
        np.testing.assert_equal(uniforms(seed('x',0),10)[5:],uniforms(seed('x',0),10)[5:])
        self.assertNotEqual(seed('x',0),seed('x',1))
    def test_repetition(self):
        self.assertEqual(r10('short text'),0)
        self.assertGreater(r10('hello '*100),.5)
    def test_jobs_full_registration(self):
        spec=json.loads(Path('round3_spec.json').read_text())
        rows=[{'uid':f'{t}:{i}','task':t,'image_sha256':f'{t}-image-{i}'} for t in TASKS for i in range(100)]
        mechanism=[r for r in rows if r['task'] in TASKS[:2] and int(r['uid'].split(':')[-1])<30]
        jobs,counts=build(spec,rows,mechanism,'test-profile')
        self.assertEqual(len(jobs),23160)
        self.assertEqual(counts,dict(main=19500,controls=2400,rescue=720,stop=540))
        self.assertTrue(all('answer' not in j for j in jobs))
    def test_rescue_matched_controls(self):
        result=masks('u',[0,1,2,3],[4],[5],list(range(10)))
        for label in ('Value','Binding'):
            self.assertEqual(result[label+'Rescue']['removed'],result[label+'Random']['removed'])
            self.assertEqual(len(result[label+'Rescue']['selected_indices']),4)
            self.assertFalse(set(result[label+'Random']['added'])&{4,5})
        self.assertEqual(result,masks('u',[0,1,2,3],[4],[5],list(range(10))))
class DiagnosticConfigTests(unittest.TestCase):
    def test_legacy_eos_fallback_disabled_and_restored_after_failure(self):
        from types import SimpleNamespace
        from r3.model_resume_probe import diagnostic_no_eos
        original=SimpleNamespace(eos_token_id=2,forced_eos_token_id=2,_from_model_config=True)
        model=SimpleNamespace(generation_config=original)
        with self.assertRaises(RuntimeError):
            with diagnostic_no_eos(model) as config:
                self.assertIsNone(model.generation_config.eos_token_id)
                self.assertIsNone(config.forced_eos_token_id)
                self.assertFalse(config._from_model_config)
                self.assertEqual(original.eos_token_id,2)
                raise RuntimeError('simulate interruption')
        self.assertIs(model.generation_config,original)

class RunnerTests(unittest.TestCase):
    def fixture(self):
        from r3.common import digest
        job={'job_key':digest('job'),'uid':'u','task':'DocVQA','image_sha256':'image','model':'llava7',
             'condition':'Full','stage':'main','generation':{'max_new_tokens':2}}
        return job,{'u':{'uid':'u','task':'DocVQA','image_sha256':'image','question':'q'}}
    def test_success_is_immutable_and_not_generated_twice(self):
        from r3.runner import run_jobs
        job,rows=self.fixture();calls=[]
        class Session:
            def generate(self,*args):
                calls.append(1)
                return dict(generated_token_ids=[5,2],output_tokens=2,output_cap_reached=True,eos_terminated=True)
        with tempfile.TemporaryDirectory() as d:
            factory=lambda *args:Session()
            self.assertEqual(run_jobs([job],rows,{'code':'a'},d,factory),1)
            path=Path(d)/job['job_key']/'result.json';original=path.read_bytes()
            self.assertEqual(run_jobs([job],rows,{'code':'a'},d,factory),1)
            self.assertEqual(calls,[1]);self.assertEqual(original,path.read_bytes())
            with self.assertRaises(Blocked):run_jobs([job],rows,{'code':'b'},d,factory)
    def test_failure_retained_and_not_automatically_resampled(self):
        from r3.runner import run_jobs
        job,rows=self.fixture();calls=[]
        class Session:
            def generate(self,row,job,prefix):
                calls.append(1);Path(prefix).write_text('original prefix')
                raise RuntimeError('simulated device failure')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):run_jobs([job],rows,{},d,lambda *a:Session())
            with self.assertRaises(Blocked):run_jobs([job],rows,{},d,lambda *a:Session())
            self.assertEqual(calls,[1]);directory=Path(d)/job['job_key']
            self.assertEqual(len(list(directory.glob('failure-*.json'))),1)
            self.assertEqual((directory/'prefix.json').read_text(),'original prefix')
    def test_unsupported_and_early_stop_fail_closed(self):
        from r3.runner import run_jobs
        job,rows=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            changed=dict(job,condition='Random25_L2')
            with self.assertRaises(Blocked):run_jobs([changed],rows,{},d,lambda *a:self.fail('must not load'))
            class Session:
                def generate(self,*a):return dict(generated_token_ids=[5],output_tokens=1,output_cap_reached=False,eos_terminated=False)
            with self.assertRaises(Blocked):run_jobs([job],rows,{},d,lambda *a:Session())
            self.assertFalse((Path(d)/job['job_key']/'result.json').exists())
    def test_atomic_publish_refuses_overwrite(self):
        from r3.runner import publish_once
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json';publish_once(p,{'v':1})
            with self.assertRaises(FileExistsError):publish_once(p,{'v':2})
            self.assertEqual(json.loads(p.read_text()),{'v':1})

class OfficialMappingTests(unittest.TestCase):
    def test_lossless_names_and_unrecognized_tensor_rejection(self):
        from r3.fastv_official_probe import mapped_name
        self.assertEqual(mapped_name('model.layers.0.self_attn.q_proj.weight'),'language_model.model.layers.0.self_attn.q_proj.weight')
        self.assertEqual(mapped_name('model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight'),'vision_tower.vision_model.embeddings.patch_embedding.weight')
        self.assertEqual(mapped_name('model.mm_projector.2.bias'),'multi_modal_projector.linear_2.bias')
        with self.assertRaises(Blocked):mapped_name('unrecognized.weight')

class ProcessorHashTests(unittest.TestCase):
    def test_only_known_version_metadata_can_differ(self):
        from r3.native_session import processor_hash_for_audit
        from r3.common import digest
        config={'size':336,'do_resize':True};expected=digest(json.dumps(config,sort_keys=True))
        extended=dict(config,_valid_processor_keys=['images','do_resize','size','resample','do_center_crop','crop_size','do_rescale','rescale_factor','do_normalize','image_mean','image_std','do_convert_rgb','return_tensors','data_format','input_data_format'])
        raw,normalized=processor_hash_for_audit(extended,expected)
        self.assertNotEqual(raw,expected);self.assertEqual(normalized,expected)
        with self.assertRaises(Blocked):processor_hash_for_audit(dict(extended,size=224),expected)
        with self.assertRaises(Blocked):processor_hash_for_audit(dict(extended,_valid_processor_keys=['anything']),expected)

class RegistrationTests(unittest.TestCase):
    def test_whole_stage_counts_and_control_gaps(self):
        from r3.register_run import compile_plans,register
        spec=Path('round3_spec.json').resolve()
        rows=[dict(uid=t+str(i),task=t,image_sha256=t+str(i)) for t in TASKS for i in range(100)]
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);manifest=folder/'manifest.jsonl';profile=folder/'profile.json'
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows));profile.write_text('{}')
            base,groups,inventory=compile_plans(spec,manifest,profile,['main'])
            self.assertEqual(inventory['planned_records'],19500)
            self.assertEqual(len(groups),13);self.assertEqual(inventory['unsupported_conditions'],[])
            self.assertIn('llava7/AgilePruner25',groups)
            _,_,controls=compile_plans(spec,manifest,profile,['controls'])
            self.assertEqual(controls['planned_records'],2400)
            self.assertIn('llava7/Random25_L2',controls['unsupported_conditions'])
            with self.assertRaises(Blocked):register(folder,spec,manifest,profile,['controls'],'absent','absent',folder/'registration')
            self.assertFalse((folder/'registration').exists())
    def test_missing_and_stale_gate_never_certify(self):
        from r3.common import require_gate,write,code_hash
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gate.json'
            write(p,{'passed':True,'spec_sha256':'s','engine_sha256':'e','code_sha256':'old'})
            with self.assertRaises(Blocked):require_gate(p,'s','e')
            write(p,{'passed':False,'spec_sha256':'s','engine_sha256':'e','code_sha256':code_hash()})
            with self.assertRaises(Blocked):require_gate(p,'s','e')

if __name__=='__main__':unittest.main()
