"""Prepare full natural certification jobs, without launching or inventing a backend.

The existing native-b1 production worker is gate-locked to main/controls. It must
not be relabelled twice as two independent natural-certification backends. This
module emits immutable preparation data and explicit missing execution adapters.
"""
import argparse
import json
from pathlib import Path
from .common import Blocked,digest,seed,read,write,code_hash
from .manifest import TASKS

CONDITIONS={
    'llava7':['Full','FastV25','PriorTR25','AgilePruner25','VisionZip25'],
    'qwen8':['Full','PriorTR25'],
    'mini8':['MiniNative'],
    'qwen4':['Full','PriorTR25'],
}
CAPS={'llava7':2048,'qwen8':32768,'mini8':2048,'qwen4':32768}


def validate_calibration(rows):
    if len(rows)!=100 or len({r['uid'] for r in rows})!=100 or len({r['image_sha256'] for r in rows})!=100:
        raise Blocked('Require all 100 unique frozen calibration images')
    if any(sum(r['task']==task for r in rows)!=20 for task in TASKS):
        raise Blocked('Require all five tasks, 20 calibration images each')
    # Keep the generation plans label-free, even if a caller supplies a scored manifest.
    if any(any(k in r for k in ('answer','answers','label','public_label')) for r in rows):
        raise Blocked('Calibration generation manifest must not contain labels')


def build(spec,rows,manifest_sha,include_qwen4=False):
    validate_calibration(rows)
    models=['llava7','qwen8','mini8']+(['qwen4'] if include_qwen4 else [])
    jobs=[];schedule=[];blockers=[]
    expected_generation={'temperature':1.0,'top_p':.95,'top_k':20,'repetition_penalty':1.0,'min_new_tokens':0}
    for field,value in expected_generation.items():
        if spec['generation'].get(field)!=value:raise Blocked('Generation protocol changed: '+field)
    order=sorted(rows,key=lambda r:(TASKS.index(r['task']),digest('r3-natural|'+r['uid'])))
    for model in models:
        if spec['models'][model]['max_new_tokens']!=CAPS[model]:raise Blocked('Natural cap changed: '+model)
        missing=[r['uid'] for r in rows if not r.get('native_inputs',{}).get(model,{}).get('eligible')]
        if missing:blockers.append({'model':model,'reason':'calibration native input audit missing','uids':missing})
        for condition in CONDITIONS[model]:
            for side in ('reference','candidate'):
                group=model+'/'+condition+'/'+side;keys=[]
                for repeat in range(3):
                    for row in order:
                        generation=dict(spec['generation'],max_new_tokens=CAPS[model])
                        identity=dict(protocol=spec['protocol'],stage='natural_certification',model=model,
                            condition=condition,side=side,uid=row['uid'],repeat=repeat,
                            manifest_sha256=manifest_sha,generation=generation)
                        key=digest(json.dumps(identity,sort_keys=True));keys.append(key)
                        jobs.append(dict(identity,job_key=key,seed=seed(row['uid'],repeat),
                            task=row['task'],image_sha256=row['image_sha256'],record_complete=False))
                schedule.append(dict(order=len(schedule),group=group,model=model,condition=condition,side=side,
                    jobs=keys,count=300,status='not_started',execution_adapter=None,
                    isolation='one native environment process, one resident model; serial single GPU'))
    blockers.extend([
        {'reason':'reference natural worker is not implemented for the full condition matrix; diagnostics only force one image'},
        {'reason':'candidate production backend/profile has not passed native/long numeric gates; current sole NativeSession cannot be duplicated as independent backends'},
        {'reason':'natural worker must use request-local uniform stream, original cap/EOS, immutable prefix recovery and complete input binding'},
        {'reason':'final-region parser and complete task scoring must supply score, reasoning boundaries and provenance before statistical certification'},
    ])
    expected=6000 if include_qwen4 else 4800
    if len(jobs)!=expected or len({j['job_key'] for j in jobs})!=expected:raise Blocked('Natural job matrix incomplete')
    return dict(stage='natural_certification_preparation',started=False,executable=False,
        production_started=False,native_gate_passed=False,planned_records=expected,
        completed_records=0,calibration_images=100,repeats=3,include_qwen4=include_qwen4,
        jobs=jobs,schedule=schedule,blockers=blockers,
        acceptance=dict(bootstrap_replicates=20000,bootstrap_seed=20261022,
            log1p_length_margin=.10,cap_probability_margin=.05,task_score_margin=.03,
            qwen_reasoning_behavior='report separately; no invented margin'),
        scope='Preparation only. Jobs may be bound to reviewed distinct paths only after native/long gates; no processes started.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--include-qwen4',action='store_true');a=p.parse_args()
    if a.out.exists():raise Blocked('Preparation artifact immutable')
    if a.calibration.name!='calibration_manifest.jsonl':raise Blocked('Use frozen calibration manifest only')
    rows=[json.loads(x) for x in a.calibration.read_text(encoding='utf-8').splitlines()]
    result=build(read(a.spec),rows,digest(a.calibration.read_bytes()),a.include_qwen4)
    result.update(spec_sha256=digest(a.spec.read_bytes()),code_sha256=code_hash(),manifest_path=str(a.calibration))
    write(a.out,result)
    print(json.dumps({k:result[k] for k in ('planned_records','started','executable','completed_records','blockers')}))

if __name__=='__main__':main()
