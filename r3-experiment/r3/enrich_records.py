"""Offline original-token partitions and final-region text; no model inference.

Input is one model's immutable JSONL records and actual prompt-ID receipts.
Output is no-replace JSONL usable by task_scoring. It preserves every input
field and original generated ID, and rejects conflicting existing metadata.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from .common import Blocked, digest, read
from .scoring import MARKER, extract
from .token_partitions import partition_generated, prompt_has_open_think


def file_digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def prompt_index(path):
    rows = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        key = (row['model'], row['uid'])
        ids = row.get('prompt_ids')
        if key in rows or not isinstance(ids,list) or not ids or any(type(t) is not int for t in ids):
            raise Blocked('Duplicate or invalid actual prompt-ID receipt')
        sha = digest(json.dumps(ids))
        if row.get('prompt_sha256',sha) != sha:
            raise Blocked('Prompt-ID receipt hash mismatch')
        rows[key] = row
    return rows


def enrich_record(record, prompt, tokenizer, eos_token_ids):
    if record.get('record_complete') is not True or record.get('error'):
        raise Blocked('Only complete immutable generation records can be enriched')
    if (record['model'],record['uid']) != (prompt['model'],prompt['uid']):
        raise Blocked('Prompt/record identity mismatch')
    prompt_ids = prompt['prompt_ids']
    if digest(json.dumps(prompt_ids)) != record.get('prompt_sha256'):
        raise Blocked('Actual prompt does not match generation receipt')
    model = record['model']
    state = prompt_has_open_think(prompt_ids,tokenizer) if model in ('qwen8','qwen4') else None
    part = partition_generated(record['generated_token_ids'],tokenizer,model,
                               record['generation']['max_new_tokens'],eos_token_ids,
                               prompt_think_open=state)
    if model in ('qwen8','qwen4'):
        scope = 'after_first_generated_think_close' if part['thinking_boundary_found'] else 'empty_unclosed_think'
        final_text = part['final_text']
    else:
        visible = tokenizer.decode(record['generated_token_ids'],skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
        markers = list(MARKER.finditer(visible))
        final_text = visible[markers[-1].start():] if markers else visible
        scope = 'last_final_answer_line_onward' if markers else 'whole_non_thinking_output_for_boxed_fallback'
    prediction, _ = extract(final_text)
    additions = dict(part,final_text=final_text,final_text_scope=scope,
                     final_region_parse_failure=prediction is None,
                     enrichment_version='r3-offline-original-token-v1',
                     original_record_sha256=digest(json.dumps(record,sort_keys=True,ensure_ascii=False)))
    for key,value in additions.items():
        if key in record and record[key] != value:
            raise Blocked('Existing output field conflicts with original-ID partition: '+key)
    return dict(record,**additions)


def enrich_file(records_path, prompts_path, out_path, model_key, tokenizer, eos_token_ids,
                tokenizer_provenance):
    records_path,prompts_path,out_path = map(Path,(records_path,prompts_path,out_path))
    if out_path.exists(): raise Blocked('Enriched output is immutable; choose a new path')
    if out_path.resolve() in (records_path.resolve(),prompts_path.resolve()):
        raise Blocked('Output must not replace original inputs')
    prompts = prompt_index(prompts_path)
    provenance = {'records_sha256':file_digest(records_path),
                  'prompt_receipts_sha256':digest(prompts_path.read_bytes()),
                  'tokenizer':tokenizer_provenance,'eos_token_ids':eos_token_ids,
                  'enricher_sha256':digest(Path(__file__).read_bytes()),
                  'partitioner_sha256':digest(Path(__file__).with_name('token_partitions.py').read_bytes())}
    out_path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary = tempfile.mkstemp(prefix=out_path.name+'.pending-',dir=out_path.parent)
    keys=set();count=0
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8',newline='\n') as output:
            with records_path.open(encoding='utf-8') as source:
                for line in source:
                    if not line.strip():continue
                    record=json.loads(line)
                    if record['model']!=model_key:raise Blocked('Mixed model input; supply one tokenizer/model per output')
                    key=record['job_key']
                    if key in keys:raise Blocked('Duplicate generation job')
                    keys.add(key)
                    prompt=prompts.get((model_key,record['uid']))
                    if prompt is None:raise Blocked('Missing actual prompt IDs; do not reconstruct from guessed text')
                    enriched=enrich_record(record,prompt,tokenizer,eos_token_ids)
                    if 'enrichment_provenance' in enriched:raise Blocked('Input was already enriched')
                    enriched['enrichment_provenance']=provenance
                    output.write(json.dumps(enriched,ensure_ascii=False)+'\n');count+=1
            if not count:raise Blocked('No generation records')
            output.flush();os.fsync(output.fileno())
        os.link(temporary,out_path)  # atomic publication; no replacement
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'records':count,'output':str(out_path),'output_sha256':file_digest(out_path)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('records','prompts','tokenizer','generation-config','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=['llava7','mini8','qwen8','qwen4'],required=True)
    a=p.parse_args()
    if not a.tokenizer.is_dir():raise Blocked('Tokenizer must be a local frozen checkpoint directory')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(a.tokenizer),use_fast=a.model.startswith('qwen'),
                                           local_files_only=True,trust_remote_code=False)
    names=('tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json',
           'added_tokens.json','vocab.json','merges.txt')
    files={name:digest((a.tokenizer/name).read_bytes()) for name in names if (a.tokenizer/name).is_file()}
    if not files:raise Blocked('No tokenizer assets to bind')
    generation=read(a.generation_config)
    if 'eos_token_id' not in generation:raise Blocked('Need actual frozen generation_config EOS IDs')
    provenance={'path':str(a.tokenizer.resolve()),'files':files,'class':type(tokenizer).__name__,
                'generation_config_sha256':digest(a.generation_config.read_bytes())}
    print(json.dumps(enrich_file(a.records,a.prompts,a.out,a.model,tokenizer,generation['eos_token_id'],provenance)))


if __name__=='__main__':main()
