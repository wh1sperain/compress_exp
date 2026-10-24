"""V3 extraction preserves punctuation/units. Scoring sees labels; workers do not."""
import re
from collections import Counter

MARKER=re.compile(r'^[^\S\r\n]*FINAL ANSWER[^\S\r\n]*[:：][^\S\r\n]*(\S[^\r\n]*?)[^\S\r\n]*$',re.I|re.M)

def extract(text):
    matches=list(MARKER.finditer(text))
    if matches: return matches[-1].group(1).strip(),'last_final_answer_line'
    # Balanced complete boxed spans only; never fall back to last number.
    candidates=[]
    for m in re.finditer(r'\\boxed\{',text):
        level=1;start=m.end()
        for i in range(start,len(text)):
            if text[i]=='{': level+=1
            elif text[i]=='}': level-=1
            if level==0:
                if text[start:i].strip(): candidates.append(text[start:i].strip())
                break
    return (candidates[-1],'complete_boxed') if candidates else (None,'missing')

def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        cur=[i]
        for j,d in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(c!=d)))
        prev=cur
    return prev[-1]

def nls(prediction,references):
    if not references: raise ValueError('Missing reference labels')
    if prediction is None:return 0.
    p=prediction.strip().lower();scores=[]
    for a in references:
        a=str(a).strip().lower();distance=edit_distance(p,a)/max(len(p),len(a),1)
        scores.append(1-distance if distance<.5 else 0.)
    return max(scores)

def r10(text):
    units=re.findall(r'\w+|[^\w\s]',text.lower())
    grams=[tuple(units[i:i+10]) for i in range(len(units)-9)]
    return 1-len(set(grams))/len(grams) if grams else 0.

def explicit_boundary(ids,tokenizer):
    text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    matches=list(MARKER.finditer(text));pred,source=extract(text)
    result={'marker_count':len(matches),'first_marker_char':matches[0].start() if matches else None,
            'last_marker_char':matches[-1].start() if matches else None,'explicit_explanation_tokens':None,
            'boundary_source':None,'prediction_v3':pred,'parse_failure':pred is None,
            'marker_followed_by_nonempty_text':bool(matches and text[matches[-1].end():].strip())}
    if not matches:return result
    boundary=matches[-1].start()
    # Incremental original-ID decoding; never count retokenized strings.
    for k in range(len(ids)+1):
        prefix=tokenizer.decode(ids[:k],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if len(prefix)>=boundary and text.startswith(prefix):
            result.update(explicit_explanation_tokens=k,boundary_source='original_ids_incremental_decode',
                          boundary_crosses_token=len(prefix)>boundary)
            return result
    raise ValueError('Non-prefix decoder: boundary cannot be mapped safely')
