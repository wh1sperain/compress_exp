"""Compile complete stage plans; production publication requires fresh gates.

Dry-run is an inventory, never a launch or a passing certification.
"""
import argparse,json
from pathlib import Path
from .common import read,digest,code_hash,Blocked,require_gate
from .build_jobs import build
from .runner import publish_once,supported
from .control_adapters import uses_fastv_runtime


def compile_plans(spec_path,manifest_path,profile_path,stages):
    spec_path=Path(spec_path);manifest_path=Path(manifest_path);profile_path=Path(profile_path)
    spec=read(spec_path)
    rows=[json.loads(s) for s in manifest_path.read_text(encoding='utf-8').splitlines() if s.strip()]
    profile_sha=digest(profile_path.read_bytes())
    jobs,counts=build(spec,rows,None,profile_sha)
    if not stages or not set(stages)<={'main','controls'}:raise Blocked('Unknown or empty stages')
    selected=[j for j in jobs if j['stage'] in stages]
    groups={}
    for job in selected:groups.setdefault(job['model']+'/'+job['condition'],[]).append(job)
    inventory={'planned_records':len(selected),'counts':{s:counts[s] for s in stages},
               'deferred_stages':['rescue','stop'],'unsupported_conditions':sorted({j['model']+'/'+j['condition'] for j in selected if not supported(j)})}
    base={'backend':'native-b1','spec_path':str(spec_path.resolve()),'spec_sha256':digest(spec_path.read_bytes()),
          'manifest_path':str(manifest_path.resolve()),'manifest_sha256':digest(manifest_path.read_bytes()),
          'engine_profile_path':str(profile_path.resolve()),'engine_sha256':profile_sha,'code_sha256':code_hash()}
    return base,groups,inventory


def register(folder,spec,manifest,profile,stages,native_gate,engine_gate,out,supplemental_audit=None,supplemental_status=None,execution_acceptance=None):
    folder=Path(folder);out=Path(out)
    base,groups,inventory=compile_plans(spec,manifest,profile,stages)
    if inventory['unsupported_conditions']:raise Blocked('Unimplemented controls: '+str(inventory['unsupported_conditions']))
    identity={'manifest_sha256':base['manifest_sha256'],
              'source_lock_sha256':digest((folder/'manifests/source_lock.json').read_bytes()),
              'weights_lock_sha256':digest((folder/'manifests/model_weights_lock.json').read_bytes())}
    if any(j['model']=='qwen4' for jobs in groups.values() for j in jobs):
        if not supplemental_audit or not supplemental_status:raise Blocked('Qwen4 supplemental audit required')
        from .supplemental_inputs import attach
        rows=[json.loads(s) for s in Path(manifest).read_text(encoding='utf-8').splitlines() if s.strip()]
        _,extra=attach(rows,supplemental_audit,supplemental_status);identity.update(extra)
        base.update(supplemental_audit_path=str(Path(supplemental_audit).resolve()),supplemental_status_path=str(Path(supplemental_status).resolve()))
    if execution_acceptance:
        if native_gate or engine_gate:raise Blocked('Choose strict gates or explicit reduced acceptance, never mix them')
        if set(stages)!={'main','controls'} or inventory['planned_records']!=21900:raise Blocked('Reduced acceptance covers complete main+controls only')
        from .execution_acceptance import validate,MODE
        bindings=dict(identity,spec_sha256=base['spec_sha256'],engine_sha256=base['engine_sha256'],code_sha256=base['code_sha256'])
        validate(execution_acceptance,bindings,groups)
        base.update(execution_acceptance_path=str(Path(execution_acceptance).resolve()),
                    execution_acceptance_sha256=digest(Path(execution_acceptance).read_bytes()),execution_mode=MODE,inference_only=True)
    else:
        if not native_gate or not engine_gate:raise Blocked('Strict mode requires both certification gates')
        for path in (native_gate,engine_gate):
            gate=require_gate(path,base['spec_sha256'],base['engine_sha256'])
            if any(gate.get(k)!=v for k,v in identity.items()):raise Blocked('Certification source/input mismatch')
            needed={key.removesuffix('_T0p6') for key in groups}
            if not needed<=set(gate.get('certified_conditions',[])):raise Blocked('Incomplete certification coverage')
    if out.exists():raise Blocked('Existing registration must remain immutable')
    out.mkdir(parents=True)
    plans=[]
    for name,jobs in groups.items():
        plan=dict(base,jobs=jobs)
        if not execution_acceptance:plan.update(native_gate_path=str(Path(native_gate).resolve()),engine_gate_path=str(Path(engine_gate).resolve()))
        path=out/(name.replace('/','-')+'.json');publish_once(path,plan)
        env='qwen' if jobs[0]['model'].startswith('qwen') else 'mini' if jobs[0]['model']=='mini8' else 'fastv' if uses_fastv_runtime(jobs[0]['condition']) else 'llava'
        plans.append({'path':str(path.resolve()),'sha256':digest(path.read_bytes()),'records':len(jobs),'environment':env})
    receipt={'plans':plans,'inventory':inventory,'identity':identity,'code_sha256':base['code_sha256'],
             'production_started':False,'gates_verified':not bool(execution_acceptance)}
    if execution_acceptance:receipt.update(execution_mode=base['execution_mode'],execution_acceptance_path=base['execution_acceptance_path'],
         execution_acceptance_sha256=base['execution_acceptance_sha256'],full_native_engine_certification=False,inference_only=True)
    publish_once(out/'registration.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--spec',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--stages',nargs='+',choices=['main','controls'],default=['main','controls'])
    p.add_argument('--native-gate');p.add_argument('--engine-gate');p.add_argument('--out');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--supplemental-audit');p.add_argument('--supplemental-status')
    p.add_argument('--execution-acceptance')
    a=p.parse_args()
    if a.dry_run:print(json.dumps(compile_plans(a.spec,a.manifest,a.profile,a.stages)[2]))
    else:
        if not a.out or (not a.execution_acceptance and not (a.native_gate and a.engine_gate)):p.error('Registration needs an output path and strict gates or explicit execution acceptance')
        register(a.folder,a.spec,a.manifest,a.profile,a.stages,a.native_gate,a.engine_gate,a.out,a.supplemental_audit,a.supplemental_status,a.execution_acceptance)
