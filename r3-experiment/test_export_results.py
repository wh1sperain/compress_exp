import hashlib
import io
import json
import tempfile
import tarfile
import unittest
from pathlib import Path
from unittest.mock import patch
from r3.common import Blocked,write
from r3.runner import canonical
from r3.export_results import verify_results,sha_file,artifact_tree,export
from local_collect_formal import download,verify_archive,safe_member,collect

class LocalSFTP:
    def file(self,path,mode):return open(path,mode)
    def stat(self,path):return Path(path).stat()


def fixture(root,n=2,completed=2):
    root=Path(root);reg=root/'registration';out=root/'formal';reg.mkdir();out.mkdir()
    jobs=[dict(job_key=hashlib.sha256(str(i).encode()).hexdigest(),stage='main',uid=str(i),generation={'max_new_tokens':8}) for i in range(n)]
    plan=reg/'llava.json';write(plan,{'jobs':jobs})
    registration=reg/'registration.json';write(registration,{'plans':[{'path':str(plan),'sha256':sha_file(plan),'records':n}], 'inventory':{'planned_records':n}})
    write(out/'binding.json',{'registration_sha256':sha_file(registration)})
    write(out/'completion.json',{'stage':'generation_complete','exit_code':0})
    for job in jobs[:completed]:
        path=out/'llava'/job['job_key'];binding={'job':job,'input_sha256':'input'}
        write(path/'binding.json',binding)
        write(path/'result.json',dict(job,record_complete=True,binding_sha256=canonical(binding),
            generated_token_ids=[11,12],output_tokens=2,output_text='full reasoning and answer',eos_terminated=True,output_cap_reached=False))
        write(path/'prefix.json',{'generated_token_ids':[11,12]})
    return registration,out,jobs

class ExportTests(unittest.TestCase):
    def test_full_correspondence_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            reg,out,jobs=fixture(d)
            result,_=verify_results(out,reg,2);self.assertTrue(result['verified_complete'])
            path=out/'llava'/jobs[0]['job_key']/'result.json';r=json.loads(path.read_text());r['output_tokens']=7;write(path,r)
            result,_=verify_results(out,reg,2);self.assertFalse(result['verified_complete']);self.assertEqual(len(result['invalid_results']),1)
    def test_duplicate_job_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            reg,out,jobs=fixture(d)
            source=out/'llava'/jobs[0]['job_key']/'result.json';target=out/'other'/jobs[0]['job_key'];target.mkdir(parents=True)
            (target/'result.json').write_bytes(source.read_bytes())
            r,_=verify_results(out,reg,2);self.assertFalse(r['verified_complete']);self.assertTrue(r['invalid_results'])
    def test_forbidden_output_is_not_packaged(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'models';p.mkdir();(p/'x.json').write_text('{}')
            with self.assertRaises(Blocked):artifact_tree(d)
    def test_snapshot_and_roundtrip_preserves_raw(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reg,out,_=fixture(root,n=21900,completed=1)
            write(out/'failure-old.json',{'original_failure':True})
            receipt=export(root,reg,out,root/'archive',snapshot=True)
            self.assertFalse(receipt['complete']);self.assertEqual(receipt['verified_results'],1)
            downloaded=root/'download'
            result=collect(LocalSFTP(),receipt['receipt_path'].replace('\\','/'),downloaded,extract=True)
            self.assertFalse(result['complete']);self.assertTrue(result['snapshot'])
            raw=list((downloaded/'contents/formal').rglob('result.json'))
            self.assertEqual(json.loads(raw[0].read_text())['output_text'],'full reasoning and answer')
            self.assertTrue((downloaded/'contents/formal/failure-old.json').exists())
    def test_resume_and_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=b'1234567890'*10000;remote=root/'remote';remote.write_bytes(data);target=root/'local'
            target.with_name('local.partial').write_bytes(data[:333])
            download(LocalSFTP(),str(remote),target,len(data),hashlib.sha256(data).hexdigest())
            self.assertEqual(target.read_bytes(),data)
            wrong=root/'bad';wrong.with_name('bad.partial').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):download(LocalSFTP(),str(remote),wrong,len(data),hashlib.sha256(data).hexdigest())
            self.assertEqual(wrong.with_name('bad.partial').read_bytes(),b'corrupt')
    def test_traversal_links_unlisted_and_hash_rejected(self):
        for name in ['../escape','/absolute','formal/../../escape','formal\\evil','formal/C:bad']:
            with self.assertRaises(ValueError):safe_member(name)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.tar.gz'
            with tarfile.open(path,'w:gz') as t:
                member=tarfile.TarInfo('formal/link');member.type=tarfile.SYMTYPE;member.linkname='/outside';t.addfile(member)
            with self.assertRaises(ValueError):verify_archive(path,{'files':[],'complete':False,'snapshot':True})

if __name__=='__main__':unittest.main()
