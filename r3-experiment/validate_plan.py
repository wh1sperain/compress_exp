#!/usr/bin/env python3
"""CPU-only validation of the R3 plan; does not run or certify model inference."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def kv_bytes_per_token(config: dict, element_bytes: int = 2) -> int:
    c = config.get('text_config', config)
    layers, heads = int(c['num_hidden_layers']), int(c['num_attention_heads'])
    kv_heads = int(c.get('num_key_value_heads', heads))
    dim = int(c.get('head_dim', int(c['hidden_size']) // heads))
    if min(layers, heads, kv_heads, dim, element_bytes) <= 0:
        raise ValueError('Invalid positive dimensions')
    if heads % kv_heads:
        raise ValueError('Query heads must be divisible by KV heads for this planner')
    return 2 * layers * kv_heads * dim * element_bytes


def validate(spec: dict) -> dict:
    errors: list[str] = []
    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)
    counts: dict[str, int] = {}
    for track in spec['main_tracks']:
        count = track['questions'] * track['repeats'] * len(track['conditions'])
        check(count == track['records'], f"Track {track['id']} count mismatch")
        check(len(set(track['conditions'])) == len(track['conditions']), 'Duplicate conditions')
        check(all(k in spec['conditions'] for k in track['conditions']), 'Unknown condition')
        counts['main_' + track['id']] = count
    counts['main_total'] = sum(counts.values())
    counts['controls'] = 0
    for control in spec['controls']:
        n = control['questions'] * control['repeats'] * len(control['conditions'])
        check(n == control['records'], f"Control {control['id']} count mismatch")
        counts['controls'] += n
    m = spec['mechanism']
    counts['rescue'] = m['questions'] * m['repeats'] * len(m['new_rescue_arms'])
    counts['stop'] = m['questions'] * m['repeats'] * len(m['forced_reasoning_budgets'])
    counts['core_with_controls'] = counts['main_total'] + counts['controls']
    counts['complete'] = counts['core_with_controls'] + counts['rescue'] + counts['stop']
    check(counts == spec['expected_records'], f'Expected record dictionary mismatch: {counts}')
    check(spec['models']['llava7']['max_expanded_prompt_tokens'] +
          spec['models']['llava7']['max_new_tokens'] <= spec['models']['llava7']['context_limit'],
          'LLaVA exceeds native context')
    for name, c in spec['conditions'].items():
        if c['method'] == 'VisionZip':
            check(c['dominant'] + c['contextual'] + c['cls_tokens'] == c['target_visual_tokens_llava'],
                  f'{name}: physical token budget mismatch')
            k = c['target_visual_tokens_llava']
            check(c['contextual'] == math.floor((k-1)*10/64 + .5), f'{name}: split mismatch')
    check(spec['models']['mini8']['allow_full_counterfactual'] is False, 'Mini invalid Full counterfactual')
    for t in spec['main_tracks']:
        if t['model'].startswith('qwen'):
            check(all(spec['conditions'][x]['method'] in {'Full','PriorTR'} for x in t['conditions']),
                  'Unrequested Qwen adaptation in main')
    check(not spec['generation']['quantization'] and not spec['generation']['kv_quantization'], 'Quantization enabled')
    check(not spec['generation']['stop_on_repetition'], 'Repetition changes stopping distribution')
    check(m['base_condition'] == 'PriorTR25' and m['prune_after_layers'] == 3, 'Mechanism still uses old adaptation')
    examples = {
        'llava7': {'num_hidden_layers':32,'num_attention_heads':32,'num_key_value_heads':32,'hidden_size':4096},
        'qwen8': {'text_config':{'num_hidden_layers':36,'num_attention_heads':32,'num_key_value_heads':8,'head_dim':128,'hidden_size':4096}},
    }
    memory = {key: kv_bytes_per_token(cfg) for key,cfg in examples.items()}
    check(memory == {'llava7':524288,'qwen8':147456}, 'KV arithmetic error')
    unresolved = ['models.' + k + '.revision' for k,v in spec['models'].items() if not v.get('revision')]
    unresolved += ['repositories.' + k + '.revision' for k,v in spec['repositories'].items() if not v.get('revision')]
    unresolved += ['Mini checkpoint-specific native query count, context, crop and prefusion configuration',
                   'GPU engine_profile and native/natural correctness gates']
    return {'protocol':spec['protocol'], 'plan_arithmetic_passed':not errors,
            'gpu_experiments_run':False, 'production_ready':False,
            'counts':counts,'kv_bytes_per_token_examples':memory,
            'visionzip_physical_budgets': {k:{x:v[x] for x in ('dominant','contextual','cls_tokens','target_visual_tokens_llava')}
                                         for k,v in spec['conditions'].items() if v['method']=='VisionZip'},
            'requires_prepare_and_validation':unresolved, 'errors':errors}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,default=Path(__file__).with_name('round3_spec.json'))
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    result=validate(json.loads(args.spec.read_text(encoding='utf-8')))
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(text,encoding='utf-8')
    print(text,end='')
    return 0 if result['plan_arithmetic_passed'] else 1
if __name__=='__main__':
    raise SystemExit(main())
