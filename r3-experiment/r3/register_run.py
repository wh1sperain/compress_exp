"""Compile complete stage plans; production publication requires fresh gates.

Dry-run is an inventory, never a launch or a passing certification.
"""
import argparse,json
from pathlib import Path
from .common import read,digest,code_hash,Blocked,require_gate
from .build_jobs import build
from .runner import publish_once,supported


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


def register(folder,spec,manifest,profile,stages,native_gate,engine_gate,out):
    folder=Path(folder);out=Path(out)
    base,groups,inventory=compile_plans(spec,manifest,profile,stages)
    if inventory['unsupported_conditions']:raise Blocked('Unimplemented controls: '+str(inventory['unsupported_conditions']))
    identity={'manifest_sha256':base['manifest_sha256'],
              'source_lock_sha256':digest((folder/'manifests/source_lock.json').read_bytes()),
              'weights_lock_sha256':digest((folder/'manifests/model_weights_lock.json').read_bytes())}
    for path in (native_gate,engine_gate):
        gate=require_gate(path,base['spec_sha256'],base['engine_sha256'])
        if any(gate.get(k)!=v for k,v in identity.items()):raise Blocked('Certification source/input mismatch')
        needed={key.removesuffix('_T0p6') for key in groups}
        if not needed<=set(gate.get('certified_conditions',[])):raise Blocked('Incomplete certification coverage')
    if out.exists():raise Blocked('Existing registration must remain immutable')
    out.mkdir(parents=True)
    plans=[]
    for name,jobs in groups.items():
        plan=dict(base,jobs=jobs,native_gate_path=str(Path(native_gate).resolve()),engine_gate_path=str(Path(engine_gate).resolve()))
        path=out/(name.replace('/','-')+'.json');publish_once(path,plan)
        env='qwen' if jobs[0]['model'].startswith('qwen') else 'mini' if jobs[0]['model']=='mini8' else 'fastv' if jobs[0]['condition'].startswith('FastV') else 'llava'
        plans.append({'path':str(path.resolve()),'sha256':digest(path.read_bytes()),'records':len(jobs),'environment':env})
    receipt={'plans':plans,'inventory':inventory,'identity':identity,'code_sha256':base['code_sha256'],
             'production_started':False,'gates_verified':True}
    publish_once(out/'registration.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--spec',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--stages',nargs='+',choices=['main','controls'],default=['main','controls'])
    p.add_argument('--native-gate');p.add_argument('--engine-gate');p.add_argument('--out');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.dry_run:print(json.dumps(compile_plans(a.spec,a.manifest,a.profile,a.stages)[2]))
    else:
        if not all((a.native_gate,a.engine_gate,a.out)):p.error('Production registration needs both gates and output path')
        register(a.folder,a.spec,a.manifest,a.profile,a.stages,a.native_gate,a.engine_gate,a.out)
