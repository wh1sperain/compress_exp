"""Offline label-side scoring. Never import this module into generation workers.

GQA uses exact answer equality from its reviewed evaluator, without VQA-style
normalization. MathVista normalization follows the reviewed author function, but extraction is study V3,
not the benchmark's GPT-assisted extraction pipeline.
"""
import argparse
import json
import math
import re
from pathlib import Path
from .common import Blocked, digest, read, write
from .scoring import extract, nls, edit_distance, r10

MATHVISTA_SOURCE='https://github.com/lupantech/MathVista/blob/main/evaluation/calculate_score.py'
CHART_SOURCE='https://github.com/vis-nlp/UniChart/blob/56dd59af1c3ebec1773fe8a4bf43694cd565130c/model/chartqa_model.py'
GQA_OFFICIAL_METRIC='https://cs.stanford.edu/people/dorarad/gqa/evaluate.html'
GQA_EVALUATOR_REVISION='e442a4c14af1b333067a6b8310c9111cad922033'
GQA_EVALUATOR_PATH='video/Video-LLaVA/videollava/eval/eval_gqa.py'
GQA_EVALUATOR_SHA256='5d5d1771a41cc5ca2d596a63ee0737dc68163502c79c51b1dfebc84e8248a4ae'
GQA_EVALUATOR_SOURCE=('https://github.com/CodeChildCZJ/PriorTR/blob/'+GQA_EVALUATOR_REVISION+'/'+GQA_EVALUATOR_PATH)


def gqa_exact(prediction, reference):
    """GQA evaluator: correct = (predicted == gold); no answer normalization.

    LLaVA's separate submission converter lowercases/removes trailing periods;
    that is not part of the official accuracy comparison and is not added here.
    Study V3 extraction happens upstream and remains explicitly identified.
    """
    return prediction is not None and prediction == reference


def mathvista_normalize(prediction, options, answer_type, precision):
    if prediction is None or not str(prediction).strip(): return None
    prediction=str(prediction).strip()
    if options:
        letters=re.findall(r'\(([a-zA-Z])\)',prediction)
        if letters: prediction=letters[0].upper()
        choices=[chr(65+i) for i in range(len(options))]
        if prediction in choices:return options[choices.index(prediction)]
        return options[min(range(len(options)),key=lambda i:edit_distance(prediction,options[i]))]
    try:
        if answer_type=='integer':return str(int(float(prediction)))
        if answer_type=='float':return str(round(float(prediction),int(precision)))
        if answer_type=='list':return str(prediction)
    except (ValueError,TypeError,OverflowError):return None
    raise Blocked('Unsupported MathVista answer_type: '+str(answer_type))


def chart_relaxed(prediction, reference):
    """Author UniChart compute_metric: numeric relative <=5%, otherwise lower exact.

No percent/unit removal or magnitude reinterpretation is added. Numeric zero
enters the author's exception equality branch after float conversion.
"""
    if prediction is None:return False
    gt=reference;pred=prediction
    try:
        gt=float(gt);pred=float(pred)
        return abs(gt-pred)/abs(gt)<=.05
    except (ValueError,TypeError,ZeroDivisionError):
        return str(gt).lower()==str(pred).lower()


def legacy_v2(text):
    """Historical Compress scoring_v2 extraction, shadow only; never primary."""
    def boxed(s):
        found=[]
        for m in re.finditer(r'\\boxed\s*\{',s):
            depth=1
            for i in range(m.end(),len(s)):
                depth+=(s[i]=='{')-(s[i]=='}')
                if depth==0:found.append(s[m.end():i]);break
        return found
    lines=[]
    for line in text.splitlines():
        candidate=line.strip().strip('$').strip()
        m=re.fullmatch(r'\\(?:text|mathrm)\{(FINAL ANSWER\s*[:：].*)\}',candidate,re.I)
        lines.append(m.group(1) if m else line)
    text='\n'.join(lines)
    matches=re.findall(r'^\s*(?:\*\*)?FINAL ANSWER\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*(.+?)\s*$',text,re.I|re.M)
    values=matches or boxed(text)
    if not values:return None
    value=values[-1].strip()
    nested=boxed(value)
    if nested:value=nested[-1]
    value=value.strip().strip('*').strip().strip('$').strip()
    value=re.sub(r'^\\(?:text|mathrm)\{([^{}]*)\}$',r'\1',value)
    value=re.sub(r'^\*\*(.*?)\*\*$',r'\1',value)
    if value.endswith('.') and not re.fullmatch(r'[+-]?\d+\.',value):value=value[:-1].rstrip()
    xml=re.fullmatch(r'<answer>\s*([^<>]+?)\s*</answer>',value,re.I)
    scalar=re.fullmatch(r'<\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Z])\s*>',value)
    return (xml or scalar).group(1).strip() if xml or scalar else value


def labels_as_list(label):
    if 'answers' in label: values=label['answers']
    elif 'answer' in label:values=[label['answer']]
    elif 'label' in label:values=[label['label']]
    else:raise Blocked('Reference missing')
    if not isinstance(values,list) or not values or any(v is None for v in values):
        raise Blocked('Invalid references')
    return [str(v) for v in values]


def score_prediction(prediction, row, label):
    task=row['task'];refs=labels_as_list(label)
    if task in ('DocVQA','InfographicVQA'):
        return dict(metric='nls',score=nls(prediction,refs),normalized_prediction=prediction,
                    exact=prediction is not None and prediction.strip().lower() in [r.strip().lower() for r in refs])
    if task=='ChartQA-human-test':
        return dict(metric='relaxed_accuracy',score=float(any(chart_relaxed(prediction,r) for r in refs)),
                    exact=prediction is not None and prediction.strip() in [r.strip() for r in refs],
                    normalized_prediction=prediction,source=CHART_SOURCE)
    if task=='MathVista-testmini':
        if len(refs)!=1:raise Blocked('MathVista must have one reference')
        normalized=mathvista_normalize(prediction,row.get('options') or [],row['answer_type'],row.get('precision'))
        return dict(metric='normalized_accuracy',score=float(normalized is not None and normalized==refs[0]),
                    normalized_prediction=normalized,source=MATHVISTA_SOURCE,
                    extraction_protocol='study V3, not official GPT answer extraction')
    if task=='GQA-balanced-val':
        if len(refs)!=1:raise Blocked('GQA must have one reference')
        return dict(metric='accuracy',score=float(gqa_exact(prediction,refs[0])),
                    exact=gqa_exact(prediction,refs[0]),normalized_prediction=prediction,
                    source=GQA_OFFICIAL_METRIC,evaluator_source=GQA_EVALUATOR_SOURCE,
                    evaluator_revision=GQA_EVALUATOR_REVISION,evaluator_sha256=GQA_EVALUATOR_SHA256,
                    extraction_protocol='study V3; exact equality, no LLaVA submission conversion or VQA normalization')
    raise Blocked('Unknown task: '+task)


def score_record(record, row, label):
    if record.get('status') in ('failed','error') or record.get('error'):
        return {'job_key':record['job_key'],'uid':row['uid'],'status':'infrastructure_error','score':None}
    if record.get('uid')!=row['uid']:raise Blocked('Output/input UID mismatch')
    # Require an explicit parser-produced final segment; do not scan thinking or
    # silently strip model tokens from raw generation in this scoring module.
    if 'final_text' not in record:raise Blocked('Missing final_text: final-region parser must run first')
    text=record['final_text']
    if not isinstance(text,str):raise Blocked('Invalid final_text')
    prediction,method=extract(text);shadow=legacy_v2(text)
    result=score_prediction(prediction,row,label)
    return dict(job_key=record['job_key'],uid=row['uid'],task=row['task'],status='scored',
                raw_final_text=text,prediction_v3=prediction,extraction_method=method,
                legacy_v2=shadow,legacy_v2_score=score_prediction(shadow,row,label)['score'],
                parse_failure=prediction is None,r10=r10(record.get('output_text',text)),**result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise Blocked('Scoring output immutable')
    rows=[json.loads(x) for x in a.manifest.read_text(encoding='utf-8').splitlines()]
    lookup={r['uid']:r for r in rows}
    if len(lookup)!=len(rows):raise Blocked('Duplicate manifest UID')
    labels=read(a.labels);keys=set();results=[]
    for line in a.records.read_text(encoding='utf-8').splitlines():
        record=json.loads(line);key=record['job_key'];uid=record['uid']
        if key in keys or uid not in lookup or uid not in labels:raise Blocked('Duplicate job or unmatched UID')
        keys.add(key)
        try: results.append(score_record(record,lookup[uid],labels[uid]))
        except Blocked as exc:results.append(dict(job_key=key,uid=uid,status='blocked',reason=str(exc),score=None))
    write(a.out,dict(records=results,complete=all(r['status']=='scored' for r in results),
        inputs_sha256={str(x):digest(x.read_bytes()) for x in (a.records,a.manifest,a.labels)},
        scorer_sha256=digest(Path(__file__).read_bytes()),
        caveat='Offline scoring only; blocked/error rows are not incorrect model answers.'))

if __name__=='__main__':main()
