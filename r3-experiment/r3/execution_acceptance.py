"""Explicit user-authorized reduced acceptance, never a certification gate.

This distinct policy permits unchanged native-B1 main+controls inference only.
It cannot claim that uncompleted native/engine certification has passed.
"""
import argparse
import json
from pathlib import Path
from .common import Blocked,read,digest,code_hash

MODE='user_reduced_native_b1'
MODE_BATCH='user_reduced_native_batch'
BATCH_FILES=('r3/llava_batch_decode.py','r3/qwen_batch_decode.py','r3/mini_batch_decode.py','r3/batch_scheduler.py')
COUNTS={'main':19500,'controls':2400,'total':21900}


def validate_batch_profile(profile):
    if profile.get('backend')!='native-padded-batch':raise Blocked('Batch acceptance requires native-padded-batch profile')
    if profile.get('memory_admission')!='full_remaining_cap_with_scratch' or float(profile.get('kv_scratch_factor',0))<3:
        raise Blocked('Batch profile lacks full-cap scratch admission')
    if not 1<=int(profile.get('max_live_requests',0))<=32:raise Blocked('Invalid batch slot bound')
    if profile.get('dtype')!='bfloat16' or profile.get('kv_dtype')!='bfloat16':raise Blocked('Batch precision changed')
    hashes=profile.get('backend_impl_sha256',{})
    if not set(BATCH_FILES)<=set(hashes):raise Blocked('Batch implementation hashes incomplete')
    root=Path(__file__).resolve().parent.parent
    for name,sha in hashes.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or digest(path.read_bytes())!=sha:raise Blocked('Batch implementation changed: '+name)
    receipts=profile.get('batch_probe_receipts',[])
    if not receipts:raise Blocked('Missing actual batch probe receipts')
    covered=set()
    for item in receipts:
        path=Path(item['path'])
        if digest(path.read_bytes())!=item['sha256']:raise Blocked('Batch probe receipt changed')
        probe=read(path)
        if probe.get('status')!='PASS':raise Blocked('Batch probe did not pass')
        model=probe.get('model','')
        family='qwen' if model.startswith('qwen') else 'mini' if model=='mini8' else 'llava' if model=='llava7' else None
        if family is None:raise Blocked('Batch probe model missing')
        decoder='r3/'+family+'_batch_decode.py'
        if probe.get('backend_impl_sha256',{}).get(decoder)==hashes[decoder]:
            covered.add(family);continue
        release=Path(item['source_release']).resolve();source=release/'r3'
        source_hash=digest(json.dumps({p.relative_to(source).as_posix():digest(p.read_bytes()) for p in sorted(source.rglob('*.py'))},sort_keys=True))
        if source_hash!=probe.get('code_sha256') or source_hash!=item.get('source_code_sha256'):
            raise Blocked('Probe source release identity changed')
        if digest((release/decoder).read_bytes())!=hashes[decoder]:raise Blocked('Probe decoder differs from execution decoder')
        covered.add(family)
    if covered!={'llava','qwen','mini'}:raise Blocked('Batch probes lack model-family coverage')
    return profile


def validate(path,bindings,required_conditions=(),profile=None):
    value=read(path)
    if value.get('mode') not in (MODE,MODE_BATCH) or value.get('full_native_engine_certification') is not False or value.get('inference_only') is not True:
        raise Blocked('Invalid reduced execution policy')
    if any(key in value for key in ('passed','certified_conditions')):
        raise Blocked('Execution acceptance must not impersonate a certification gate')
    if value.get('expected_records')!=COUNTS or value.get('stages')!=['main','controls']:
        raise Blocked('Reduced execution policy changed research scope')
    if not value.get('authorization_text') or digest(value['authorization_text'])!=value.get('authorization_sha256'):
        raise Blocked('Missing or changed explicit user authorization')
    if not value.get('uncompleted_certification_scope'):
        raise Blocked('Uncompleted certification must remain explicit')
    if value.get('bindings')!=bindings:
        raise Blocked('Reduced execution source/spec/profile/input binding changed')
    if bindings.get('code_sha256')!=code_hash():raise Blocked('Code changed after reduced acceptance')
    conditions=value.get('allowed_conditions',[])
    if len(conditions)!=len(set(conditions)) or not set(required_conditions)<=set(conditions):
        raise Blocked('Condition outside explicit execution scope')
    evidence=value.get('evidence',[])
    if not evidence:raise Blocked('Reduced acceptance requires actual diagnostic evidence references')
    for item in evidence:
        if digest(Path(item['path']).read_bytes())!=item['sha256']:
            raise Blocked('Diagnostic evidence changed')
    if value['mode']==MODE_BATCH:
        if profile is None:raise Blocked('Batch acceptance requires bound profile')
        validate_batch_profile(profile)
        if value.get('backend')!='native-padded-batch':raise Blocked('Batch acceptance backend mismatch')
    elif value.get('backend')!='native-b1' or (profile is not None and profile.get('backend')!='native-b1'):
        raise Blocked('B1 acceptance backend mismatch')
    return value


def create(folder,spec,manifest,profile,supplemental_audit,supplemental_status,
           evidence,authorization_text,uncompleted,out):
    from .register_run import compile_plans
    from .supplemental_inputs import attach
    from .runner import publish_once
    folder=Path(folder)
    base,groups,inventory=compile_plans(spec,manifest,profile,['main','controls'])
    if inventory['planned_records']!=21900 or inventory['unsupported_conditions']:
        raise Blocked('Incomplete or unsupported main+controls inventory')
    rows=[json.loads(s) for s in Path(manifest).read_text(encoding='utf-8').splitlines() if s.strip()]
    _,extra=attach(rows,supplemental_audit,supplemental_status)
    bindings={k:base[k] for k in ('manifest_sha256','spec_sha256','engine_sha256','code_sha256')}
    bindings.update(source_lock_sha256=digest((folder/'manifests/source_lock.json').read_bytes()),
                    weights_lock_sha256=digest((folder/'manifests/model_weights_lock.json').read_bytes()),**extra)
    profile_value=read(profile)
    mode=MODE_BATCH if base['backend']=='native-padded-batch' else MODE
    if mode==MODE_BATCH:validate_batch_profile(profile_value)
    value={'mode':mode,'full_native_engine_certification':False,'inference_only':True,
           'stages':['main','controls'],'expected_records':COUNTS,'bindings':bindings,
           'allowed_conditions':sorted(groups),'authorization_text':authorization_text,
           'authorization_sha256':digest(authorization_text),'uncompleted_certification_scope':list(uncompleted),
           'evidence':[{'path':str(Path(p).resolve()),'sha256':digest(Path(p).read_bytes())} for p in evidence],
           'backend':base['backend'],'analysis_policy':'Answer parsing and scoring disabled by user; preserve raw results for download.',
           'scientific_limitation':'User-reduced execution acceptance; full native/engine certification is not established.'}
    if not authorization_text.strip() or not uncompleted or not evidence:raise Blocked('Explicit authorization, limitations, and evidence are required')
    publish_once(out,value)
    validate(out,bindings,groups,profile_value)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('folder','spec','manifest','profile','supplemental-audit','supplemental-status','authorization-file','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--evidence',nargs='+',required=True)
    p.add_argument('--uncompleted-certifications',nargs='+',required=True)
    a=p.parse_args();value=create(a.folder,a.spec,a.manifest,a.profile,a.supplemental_audit,a.supplemental_status,
                                a.evidence,Path(a.authorization_file).read_text(encoding='utf-8'),a.uncompleted_certifications,a.out)
    print(json.dumps({'mode':value['mode'],'expected_records':value['expected_records'],'full_native_engine_certification':False}))


if __name__=='__main__':main()
