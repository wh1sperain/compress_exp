"""Attach a completed Qwen4 audit without altering frozen main/calibration sets."""
import json
from pathlib import Path
from .common import read,digest,Blocked


def attach(rows,path,status_path):
    path=Path(path);status=read(status_path)
    if status.get('stage')!='complete' or status.get('exit_code')!=0:
        raise Blocked('Supplemental audit incomplete')
    sha=digest(path.read_bytes())
    if status.get('audit_sha256')!=sha:raise Blocked('Supplemental audit hash changed')
    records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(records)!=status.get('completed') or len(records)!=status.get('total'):
        raise Blocked('Supplemental count mismatch')
    index={}
    for record in records:
        if record['model']!='qwen4' or record['uid'] in index:raise Blocked('Unexpected or duplicate supplemental record')
        audit=record['audit']
        if not audit.get('eligible') or not 0<audit['expanded_prompt_tokens']<=8192:
            raise Blocked('Supplemental input outside frozen policy')
        for key in ('processor_sha256','tokenizer_sha256','chat_template_sha256','prompt_sha256'):
            if not audit.get(key):raise Blocked('Incomplete supplemental provenance')
        index[record['uid']]=audit
    result=[]
    for row in rows:
        if row['uid'] not in index:raise Blocked('Missing supplemental input: '+row['uid'])
        if 'qwen4' in row['native_inputs'] and row['native_inputs']['qwen4']!=index[row['uid']]:
            raise Blocked('Conflicting supplemental audit')
        result.append(dict(row,native_inputs=dict(row['native_inputs'],qwen4=index[row['uid']])))
    return result,{'supplemental_input_sha256':sha,'supplemental_status_sha256':digest(Path(status_path).read_bytes())}
