"""Partition ORIGINAL token IDs. No retokenized-text length substitutes."""
import argparse
from .common import *

def partition(ids, prompt_ids, start, end, eos):
    if not isinstance(start,int) or not isinstance(end,int) or start==end: raise ValueError('Verify tokenizer special IDs')
    stop=len(ids)-(1 if ids and ids[-1] in eos else 0)
    body=ids[:stop];opens=[i for i,t in enumerate(body) if t==start];closes=[i for i,t in enumerate(body) if t==end]
    prompt_open=bool(prompt_ids and start in prompt_ids and (end not in prompt_ids or len(prompt_ids)-1-prompt_ids[::-1].index(start)>len(prompt_ids)-1-prompt_ids[::-1].index(end)))
    begin=opens[0]+1 if opens else 0
    status='ok'
    if not closes: status='missing_think_end'
    elif len(closes)!=1 or len(opens)>1 or (opens and opens[0]>closes[0]): status='ambiguous_boundary'
    elif not opens and not prompt_open: status='missing_think_start'
    boundary=closes[0] if closes else len(body)
    reasoning=[t for t in body[begin:boundary] if t not in (start,end)]
    answer=body[boundary+1:] if status=='ok' else []
    return dict(reasoning_token_ids=reasoning,answer_token_ids=answer,
                reasoning_tokens=len(reasoning) if status in ('ok','missing_think_end') and (opens or prompt_open) else None,
                answer_tokens=len(answer),total_output_tokens=len(ids),think_parse_status=status,
                final_answer_present=bool(answer),structural_tokens=len(ids)-len(reasoning)-len(answer))

def repeated_tail(ids):
    """Any >=80-token thrice repeated non-overlapping fragment implies an 80-token one."""
    tail=ids[-2048:];seen={}
    for i in range(max(0,len(tail)-79)):
        key=tuple(tail[i:i+80]);positions=seen.setdefault(key,[])
        if not positions or i>=positions[-1]+80: positions.append(i)
        if len(positions)>=3: return dict(flag=True,fragment_tokens=80,positions=positions[:3],tail_offset=max(0,len(ids)-2048))
    return dict(flag=False)

def parse(record, tokenizer):
    ids=record['raw_generated_token_ids'];vocab=tokenizer.get_vocab()
    if '<think>' not in vocab or '</think>' not in vocab: raise ValueError('Unknown thinking tokenizer')
    x=partition(ids,record['prompt_token_ids'],vocab['<think>'],vocab['</think>'],record['eos_token_ids'])
    return dict(record,**x,reasoning_trace=tokenizer.decode(x['reasoning_token_ids'],skip_special_tokens=False),
                final_answer_text=tokenizer.decode(x['answer_token_ids'],skip_special_tokens=False),
                repetition_warning=repeated_tail(ids),raw_sha256=digest(record))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True);save_rows(a.out,(parse(r,t) for r in rows(a.raw)))
