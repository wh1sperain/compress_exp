"""Fail-closed paired natural-trajectory statistics; never issues a native gate.

Input rows are externally scored calibration results, not research answers.
Require the frozen 100 images and all three seeds on both sides per condition.
"""
import math
import numpy as np
from .common import Blocked, seed
from .manifest import TASKS


def evaluate(rows, calibration, model, condition, replicates=20000):
    if replicates < 20000:
        raise Blocked('Certification requires at least 20000 bootstrap replicates')
    manifest = {r['uid']: r for r in calibration}
    if len(manifest) != 100 or len({r['image_sha256'] for r in calibration}) != 100:
        raise Blocked('Require 100 unique calibration images and UIDs')
    if any(sum(r['task'] == t for r in calibration) != 20 for t in TASKS):
        raise Blocked('Require 20 calibration images per task')
    index = {}
    for r in rows:
        if r['model'] != model or r['condition'] != condition:
            raise Blocked('Mixed condition input')
        key = (r['uid'], r['repeat'], r['side'])
        if key in index or r['uid'] not in manifest or r['repeat'] not in range(3) or r['side'] not in ('reference', 'candidate'):
            raise Blocked('Duplicate or unexpected calibration observation')
        m = manifest[r['uid']]
        if any(r[f] != m[f] for f in ('task', 'image_sha256')) or r['seed'] != seed(r['uid'], r['repeat']):
            raise Blocked('Calibration identity mismatch')
        if not r.get('record_complete') or type(r['output_cap_reached']) is not bool:
            raise Blocked('Incomplete calibration observation')
        if not isinstance(r['output_tokens'], int) or r['output_tokens'] < 0:
            raise Blocked('Invalid output length')
        if not math.isfinite(r['score']) or not 0 <= r['score'] <= 1:
            raise Blocked('Missing or invalid task score')
        index[key] = r
    if len(index) != 600:
        raise Blocked('Require 100 images x 3 seeds x 2 sides')
    fields = ['log1p_output_length', 'cap_probability', 'task_score']
    margins = [.10, .05, .03]
    if model in ('qwen8', 'qwen4'):
        fields += ['log1p_reasoning_length', 'think_end_probability']
        # Protocol requires these reported, but gives no separate acceptance
        # margins. Do not invent thresholds or silently certify them.
    grouped = {t: [] for t in TASKS}
    for uid, m in sorted(manifest.items()):
        diffs = []
        for repeat in range(3):
            a, b = [index[(uid, repeat, s)] for s in ('reference', 'candidate')]
            if a['prompt_sha256'] != b['prompt_sha256']:
                raise Blocked('Prompt differs between backends')
            v = [math.log1p(b['output_tokens']) - math.log1p(a['output_tokens']),
                 int(b['output_cap_reached']) - int(a['output_cap_reached']), b['score'] - a['score']]
            if model in ('qwen8', 'qwen4'):
                for r in (a, b):
                    if type(r.get('reasoning_tokens')) is not int or not 0 <= r['reasoning_tokens'] <= r['output_tokens'] or type(r.get('think_ended')) is not bool or r.get('boundary_anomaly'):
                        raise Blocked('Unresolved reasoning boundary')
                v += [math.log1p(b['reasoning_tokens']) - math.log1p(a['reasoning_tokens']), int(b['think_ended']) - int(a['think_ended'])]
            diffs.append(v)
        grouped[m['task']].append(np.mean(diffs, axis=0))
    rng = np.random.Generator(np.random.PCG64(20261022))
    task_reports = {}; boot = []; points = []
    for task, values in grouped.items():
        values = np.asarray(values); draws = np.empty((replicates, len(fields)))
        for start in range(0, replicates, 250):
            n = min(250, replicates-start)
            draws[start:start+n] = values[rng.integers(0, 20, (n, 20))].mean(axis=1)
        ci = np.quantile(draws, [.025, .975], axis=0).T
        task_reports[task] = {'mean': values.mean(axis=0).tolist(), 'ci95': ci.tolist()}
        boot.append(draws); points.append(values.mean(axis=0))
    ci = np.quantile(np.mean(boot, axis=0), [.025, .975], axis=0).T
    # Quality remains task-specific: distinct task score scales are not pooled.
    checks = {f: bool(ci[i, 0] > -m and ci[i, 1] < m)
              for i, (f, m) in enumerate(zip(fields, margins)) if f != 'task_score'}
    for task, report in task_reports.items():
        lo, hi = report['ci95'][2]
        checks['task_score/'+task] = bool(lo > -.03 and hi < .03)
    return {'natural_statistics_passed': all(checks.values()), 'native_gate_passed': False,
            'qwen_reasoning_behavior_review_required': model in ('qwen8', 'qwen4'),
            'model': model, 'condition': condition, 'checks': checks,
            'length_behavior': {f: {'mean': float(np.mean(points, axis=0)[i]), 'ci95': ci[i].tolist()}
                                for i, f in enumerate(fields) if f != 'task_score'},
            'per_task': task_reports, 'fields': fields, 'replicates': replicates,
            'bootstrap_seed': 20261022, 'rows': len(index),
            'scope': 'Natural statistics only; source, numeric, long-context and engine gates remain mandatory.'}
