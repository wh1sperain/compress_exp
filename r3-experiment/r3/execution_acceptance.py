"""Explicit user-authorized reduced acceptance, never a certification gate.

This distinct policy permits unchanged native-B1 main+controls inference only.
It cannot claim that uncompleted native/engine certification has passed.
"""
import argparse
import json
from pathlib import Path
from .common import Blocked,read,digest,code_hash

MODE='user_reduced_native_b1'
COUNTS={'main':19500,'controls':2400,'total':21900}


def validate(path,bindings,required_conditions=()):
    value=read(path)
    if value.get('mode')!=MODE or value.get('full_native_engine_certification') is not False or value.get('inference_only') is not True:
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
    value={'mode':MODE,'full_native_engine_certification':False,'inference_only':True,
           'stages':['main','controls'],'expected_records':COUNTS,'bindings':bindings,
           'allowed_conditions':sorted(groups),'authorization_text':authorization_text,
           'authorization_sha256':digest(authorization_text),'uncompleted_certification_scope':list(uncompleted),
           'evidence':[{'path':str(Path(p).resolve()),'sha256':digest(Path(p).read_bytes())} for p in evidence],
           'backend':'native-b1','analysis_policy':'Answer parsing and scoring disabled by user; preserve raw results for download.',
           'scientific_limitation':'User-reduced execution acceptance; full native/engine certification is not established.'}
    if not authorization_text.strip() or not uncompleted or not evidence:raise Blocked('Explicit authorization, limitations, and evidence are required')
    publish_once(out,value)
    validate(out,bindings,groups)
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
