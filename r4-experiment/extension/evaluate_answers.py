"""Blind two-step extraction/scoring. Ambiguity stays unresolved, never random guesses.

MathVista/MMMU functions are loaded from hash-locked author files. MathVerse
requires a separately versioned blind judge receipt; this script makes no API calls.
"""
import argparse
import ast
import logging
import math
import re
import random
from .common import *

def official_functions(path,expected_sha,names):
    raw=Path(path).read_bytes()
    if digest(raw)!=expected_sha:raise ValueError('Official evaluator changed')
    tree=ast.parse(raw.decode('utf-8-sig'))
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
    if {n.name for n in selected}!=set(names):raise ValueError('Official evaluator function set differs')
    import numpy as np
    class NoGuess:
        def choice(self,*a,**kw):raise ValueError('Random fallback needs blind review')
    env=dict(re=re,np=np,random=NoGuess(),logging=logging)
    if 'get_most_similar' in names:
        # Explicit letter/exact choice text already resolved by extraction.
        # Do not let official edit-distance fallback force an ambiguous answer.
        def reject_fuzzy(*args):raise ValueError('Fuzzy option fallback needs blind review')
        env['distance']=reject_fuzzy
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),env)
    return env

def extract(text,choices):
    text=text.strip()
    marked=re.findall(r'(?im)^\s*FINAL ANSWER\s*:\s*(.+?)\s*$',text)
    if marked:
        if len(set(marked))!=1:return None,'conflicting_final_answers'
        text=marked[0].strip()
    if choices:
        match=re.fullmatch(r'\(?([A-Z])\)?[.\s]*',text)
        if match and ord(match[1])-65<len(choices):return match[1],'explicit_choice'
        hits=[i for i,c in enumerate(choices) if text.casefold()==str(c).strip().casefold()]
        if len(hits)==1:return chr(65+hits[0]),'exact_choice_text'
        return None,'ambiguous_choice'
    if '\n' not in text and len(text)<=120 and text:
        return text,'short_final_answer'
    return None,'requires_blind_extraction'

def relaxed(pred,gold):
    def number(s):
        try:
            s=str(s).strip();v=float(s[:-1])/100 if s.endswith('%') else float(s)
            return v if math.isfinite(v) else None
        except (ValueError,TypeError):return None
    p=number(pred);g=number(gold)
    if p is not None and g is not None and g!=0:return abs(p-g)/abs(g)<=.05
    return str(pred).strip().casefold()==str(gold).strip().casefold()

def evaluate(records,manifest,labels,reviewed=None,official=None):
    meta={r['sample_id']:r for r in manifest};gold={r['sample_id']:r for r in labels}
    reviewed=reviewed or {};official=official or {};result=[];queue=[]
    for r in records:
        row=meta[r['sample_id']];lab=gold[r['sample_id']]
        if row['image_hash']!=r['image_hash'] or lab['image_hash']!=r['image_hash']:raise ValueError('Scoring image binding changed')
        blind_id=digest([r['sample_id'],r['raw_sha256']]);judge=reviewed.get(blind_id)
        base=dict(r,parse_success=False,predicted_answer=None,correct=None,judge_type=None,judge_version=None,judge_reason=None)
        if r.get('run_status')!='success':base.update(judge_reason='process_failure_not_an_answer')
        elif r['think_parse_status']=='missing_think_end' or (r['think_parse_status']=='ok' and not r['final_answer_text'].strip()):
            base.update(correct=False,judge_type='unfinished_policy',judge_version='v1',judge_reason='No final answer; retained in denominator')
        elif r['think_parse_status']!='ok':base.update(judge_reason='boundary_ambiguity_requires_review')
        else:
            pred,reason=extract(r['final_answer_text'],row['choices']);base.update(predicted_answer=pred,parse_success=pred is not None,judge_reason=reason)
            d=row['dataset'];g=lab['gold_answer']
            try:
                if pred is not None:
                    if d=='ChartQA':base['correct']=any(relaxed(pred,x) for x in (g if isinstance(g,list) else [g]));base['judge_version']='ChartQA relaxed-5percent compatible-v1'
                    elif d in ('GQA','AI2D'):
                        # Sources must normalize AI2D gold to an option letter, GQA
                        # retains its reference string. No fuzzy semantic matching.
                        base['correct']=pred.strip().lower().rstrip('.')==str(g).strip().lower().rstrip('.')
                        base['judge_version']='exact-match-v1'
                    elif d=='MathVista':
                        cfg=official[d];f=official_functions(cfg['local_path'],cfg['sha256'],['get_most_similar','normalize_extracted_answer','safe_equal'])
                        normalized=f['normalize_extracted_answer'](pred,row['choices'],row['question_type'],lab['answer_type'],lab.get('precision'),True)
                        base['correct']=f['safe_equal'](normalized,g);base['judge_version']=cfg['revision']
                    elif d=='MMMU':
                        cfg=official[d];f=official_functions(cfg['local_path'],cfg['sha256'],cfg['functions'])
                        if row['choices']:base['correct']=f['eval_multi_choice'](g,pred)
                        else:base['correct']=f['eval_open'](g,f['parse_open_response'](pred))
                        base['judge_version']=cfg['revision']
                    else:base['judge_reason']='MathVerse official independent judge/manual review required'
                    if base['correct'] is not None:base.update(correct=bool(base['correct']),judge_type='deterministic_rule')
            except (KeyError,ValueError,ImportError) as exc:base['judge_reason']='official_adapter_unavailable: '+str(exc)
        if judge:
            if judge.get('raw_sha256')!=r['raw_sha256'] or judge.get('sample_id')!=r['sample_id'] or not judge.get('blind') or not judge.get('judge_version') or not judge.get('judge_reason'):
                raise ValueError('Unbound/unblinded review')
            if type(judge.get('correct')) is not bool or not judge.get('judge_type'):raise ValueError('Incomplete review')
            for k in ('predicted_answer','correct','judge_type','judge_version','judge_reason'):base[k]=judge[k]
            base['parse_success']=judge['predicted_answer'] is not None
        if base['correct'] is None:
            queue.append(dict(blind_id=blind_id,sample_id=r['sample_id'],dataset=row['dataset'],question=row['question'],
                              choices=row['choices'],final_answer_text=r['final_answer_text'],raw_sha256=r['raw_sha256'],
                              review_reason=base['judge_reason']))
        result.append(base)
    return result,queue

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--inputs',required=True);p.add_argument('--labels',required=True)
    p.add_argument('--reviewed');p.add_argument('--official');p.add_argument('--out',required=True);a=p.parse_args()
    result,queue=evaluate(rows(a.records),rows(a.inputs),rows(a.labels),read(a.reviewed) if a.reviewed else None,read(a.official) if a.official else None)
    save_rows(Path(a.out)/'scored.jsonl',result);save_rows(Path(a.out)/'blind_extraction_queue.jsonl',queue)
    save(Path(a.out)/'receipt.json',dict(total=len(result),unresolved=len(queue),labels_sent_to_extractor=False,api_calls=0))
