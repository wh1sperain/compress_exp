import argparse
from collections import defaultdict
import numpy as np
from .common import *
from .statistical_analysis import ctir,bootstrap,macro_bootstrap,wilson

def summarize(pairs,nboot=10000):
    known=[(b,c) for b,c in pairs if b.get('reasoning_tokens') is not None and c.get('reasoning_tokens') is not None]
    b=np.array([x['reasoning_tokens'] for x,y in known]);c=np.array([y['reasoning_tokens'] for x,y in known])
    valid=b>=32;ratios=c[valid]/b[valid] if valid.any() else np.array([])
    quality=all(type(x.get('correct')) is bool and type(y.get('correct')) is bool for x,y in pairs)
    result=dict(n=len(pairs),length_pairs=len(known),missing_length_pairs=len(pairs)-len(known),
                ctir=ctir(b,c) if len(known)==len(pairs) else None,
                ratio_denominator=int(valid.sum()),excluded_short_baseline=int((b<32).sum()),
                short_baseline_delta_mean=float(np.mean((c-b)[b<32])) if (b<32).any() else None,
                mean_reasoning_full=float(np.mean(b)) if len(b) else None,
                mean_reasoning_compressed=float(np.mean(c)) if len(c) else None,
                ratio_percentiles=dict(zip(('median','p75','p90','p95'),np.percentile(ratios,[50,75,90,95]).tolist())) if len(ratios) else None,
                growth={str(k):dict(count=int((ratios>=k).sum()),rate=float(np.mean(ratios>=k)) if len(ratios) else None,
                                   ci95=wilson(int((ratios>=k).sum()),len(ratios))) for k in (1.5,2,3,5)},
                cap_rate_full=sum(x['total_output_tokens']>=32768 for x,y in pairs)/len(pairs),
                cap_rate=sum(y['total_output_tokens']>=32768 for x,y in pairs)/len(pairs),
                completion_rate=sum(y['finish_reason']=='eos' and y['think_parse_status']=='ok' for x,y in pairs)/len(pairs),
                accuracy_full=sum(x['correct'] for x,y in pairs)/len(pairs) if quality else None,
                accuracy=sum(y['correct'] for x,y in pairs)/len(pairs) if quality else None,
                unresolved_scores=sum(x.get('correct') is None or y.get('correct') is None for x,y in pairs),
                actual_retention_mean=float(np.mean([y['keep_ratio_actual'] for x,y in pairs])),
                visual_removed_mean=float(np.mean([y['visual_tokens_removed'] for x,y in pairs])))
    result['accuracy_delta_pp']=100*(result['accuracy']-result['accuracy_full']) if quality else None
    result['normal_completed_length_mean']=float(np.mean([y['reasoning_tokens'] for x,y in known if y['finish_reason']=='eos' and y['think_parse_status']=='ok'])) if any(y['finish_reason']=='eos' and y['think_parse_status']=='ok' for x,y in known) else None
    result['capped_observed_length_mean']=float(np.mean([y['reasoning_tokens'] for x,y in known if y['total_output_tokens']>=32768])) if any(y['total_output_tokens']>=32768 for x,y in known) else None
    result['capped_with_final_answer']=sum(y['total_output_tokens']>=32768 and y.get('final_answer_present',False) for x,y in pairs)
    result['censoring_warning']=max(result['cap_rate'],result['cap_rate_full'])>.02
    result['supplement_required']='separate higher-cap sensitivity plan; no automatic rerun' if result['censoring_warning'] else None
    if len(known)==len(pairs):result.update(bootstrap(b,c,[int(x['correct']) for x,y in pairs] if quality else None,
                                                      [int(y['correct']) for x,y in pairs] if quality else None,n=nboot))
    return result

def aggregate(records,inputs,nboot=10000,expected_jobs=None):
    if len({r['run_id'] for r in records})!=len(records):raise ValueError('Duplicate run result')
    if any(r.get('run_status')!='success' or r.get('disabled') or r.get('diagnostic') for r in records):raise ValueError('Diagnostic/failed/disabled record not a formal outcome')
    if expected_jobs is not None and {r['run_id'] for r in records}!={j['run_id'] for j in expected_jobs}:raise ValueError('Expected job matrix incomplete; retain failures and export a separately labeled snapshot')
    meta={r['sample_id']:r for r in inputs};baselines={};groups=defaultdict(list)
    def key(r):return (r['phase'],r['model'],r['dataset'],r['decoding_mode'],r['seed'],r['sample_id'])
    for r in records:
        if r['method']=='Full':
            if key(r) in baselines:raise ValueError('Full baseline repeated')
            baselines[key(r)]=r
    for r in records:
        if r['method']=='Full':continue
        if key(r) not in baselines:raise ValueError('Missing matched Full')
        b=baselines[key(r)]
        for field in ('image_hash','prompt_hash','model_revision','decoding_config','tokenizer_sha256'):
            if r[field]!=b[field]:raise ValueError('Invalid pair: '+field)
        k=(r['phase'],r['model'],r['dataset'],r['decoding_mode'],r['seed'],r['method'],r['keep_ratio_target'])
        groups[k].append((b,r))
    output=[];macro_groups=defaultdict(list)
    for k,pairs in sorted(groups.items()):
        expected={sid for phase,model,dataset,mode,seed,sid in baselines if (phase,model,dataset,mode,seed)==k[:5]}
        if {r['sample_id'] for b,r in pairs}!=expected:raise ValueError('Incomplete configuration; do not silently drop failed/missing jobs')
        row=dict(zip(('phase','model','dataset','decoding_mode','seed','method','keep_ratio_target'),k))
        row.update(summarize(pairs,nboot));row['correctness_groups']={}
        for group in ('CC','CW','WC','WW'):
            subset=[(b,c) for b,c in pairs if type(b.get('correct')) is bool and type(c.get('correct')) is bool and
                    ('C' if b['correct'] else 'W')+('C' if c['correct'] else 'W')==group]
            row['correctness_groups'][group]=dict(summarize(subset,nboot),conditional_analysis=True) if subset else dict(n=0,conditional_analysis=True)
        if k[2]=='ChartQA':
            row['chartqa_subsets']={s:summarize([(b,c) for b,c in pairs if meta[c['sample_id']]['subset']==s],nboot)
                                    for s in ('Human','Augmented') if any(meta[c['sample_id']]['subset']==s for b,c in pairs)}
        row['actual_retention_bins']={}
        for lo,hi in ((0,.15),(.15,.375),(.375,.625),(.625,.825),(.825,1.01)):
            subset=[(b,c) for b,c in pairs if lo<=c['keep_ratio_actual']<hi]
            if subset:row['actual_retention_bins'][f'{lo}-{hi}']=summarize(subset,nboot)
        if all(b.get('reasoning_tokens') is not None and c.get('reasoning_tokens') is not None for b,c in pairs):
            x=[c['visual_tokens_removed'] for b,c in pairs];y=[c['reasoning_tokens']-b['reasoning_tokens'] for b,c in pairs]
            row['removed_delta_pearson']=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
        output.append(row);macro_groups[k[:2]+k[3:]].append((row,pairs))
    macros=[]
    for k,items in macro_groups.items():
        entry=dict(zip(('phase','model','decoding_mode','seed','method','keep_ratio_target'),k))
        complete={r['dataset'] for r,p in items}==set(DATASETS) and all(r['ctir'] is not None for r,p in items)
        entry.update(tasks_present=len(items),macro_ctir=float(np.mean([r['ctir'] for r,p in items])) if complete else None,
                     primary=k[-2] in ('PriorTR','FastV') and k[-1]==.25,
                     macro_accuracy=float(np.mean([r['accuracy'] for r,p in items])) if complete and all(r['accuracy'] is not None for r,p in items) else None,
                     ci95=macro_bootstrap([([b['reasoning_tokens'] for b,c in p],[c['reasoning_tokens'] for b,c in p]) for r,p in items],n=nboot) if complete else None)
        macros.append(entry)
    seeds=defaultdict(list)
    for r in output:
        if r['decoding_mode']=='sampling':seeds[(r['phase'],r['model'],r['dataset'],r['method'],r['keep_ratio_target'])].append(r)
    robustness=[]
    for k,rr in seeds.items():
        complete={r['seed'] for r in rr}=={42,43,44}
        robustness.append(dict(configuration=k,seeds_present=[r['seed'] for r in rr],
            ctir_mean=float(np.mean([r['ctir'] for r in rr])) if complete and all(r['ctir'] is not None for r in rr) else None,
            ctir_std=float(np.std([r['ctir'] for r in rr],ddof=1)) if complete and all(r['ctir'] is not None for r in rr) else None))
        for field in ('accuracy','accuracy_delta_pp'):
            values=[r[field] for r in rr]
            robustness[-1][field+'_mean']=float(np.mean(values)) if complete and all(v is not None for v in values) else None
            robustness[-1][field+'_std']=float(np.std(values,ddof=1)) if complete and all(v is not None for v in values) else None
        values=[r['growth']['2']['rate'] for r in rr]
        robustness[-1]['twofold_rate_mean']=float(np.mean(values)) if complete and all(v is not None for v in values) else None
        robustness[-1]['twofold_rate_std']=float(np.std(values,ddof=1)) if complete and all(v is not None for v in values) else None
    bg=defaultdict(list)
    for key,value in baselines.items():bg[key[:-1]].append(value)
    baseline_summary=[]
    for k,rr in sorted(bg.items()):
        lengths=[r['reasoning_tokens'] for r in rr if r.get('reasoning_tokens') is not None]
        quality=all(type(r.get('correct')) is bool for r in rr)
        baseline_summary.append(dict(zip(('phase','model','dataset','decoding_mode','seed'),k)) |
            dict(n=len(rr),mean_reasoning=float(np.mean(lengths)) if lengths else None,
                 p95_reasoning=float(np.percentile(lengths,95)) if lengths else None,
                 accuracy=sum(r['correct'] for r in rr)/len(rr) if quality else None,
                 accuracy_ci95=wilson(sum(r['correct'] for r in rr),len(rr)) if quality else None,
                 cap_rate=sum(r['total_output_tokens']>=32768 for r in rr)/len(rr)))
    return dict(configurations=output,baselines=baseline_summary,macro=macros,robustness=robustness,
                expected_matrix_verified=expected_jobs is not None,
                note='Observed capped reasoning lengths; ratio eligibility Full >=32; CC conditional; seeds never treated as independent questions')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--inputs',required=True);p.add_argument('--jobs',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();save(a.out,aggregate(rows(a.records),rows(a.inputs),expected_jobs=rows(a.jobs)))
