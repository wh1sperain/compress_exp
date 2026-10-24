"""CPU-only frozen-tokenizer partition checks and existing-natural-record replay.

Loads tokenizer assets, never model weights or GPU. Synthetic text cases are
explicitly tokenizer tests, not real answers or a native model gate.
"""
import argparse
import json
from pathlib import Path
from .common import Blocked,digest,read
from .enrich_records import enrich_record
from .runner import publish_once
from .token_partitions import partition_generated


def synthetic_checks(tokenizer,model_key,eos_token_ids):
    eos=[] if eos_token_ids is None else [eos_token_ids] if isinstance(eos_token_ids,int) else list(eos_token_ids)
    if not eos:raise Blocked('Actual generation config has no EOS token')
    encode=lambda text:list(tokenizer.encode(text,add_special_tokens=False))
    qwen=model_key.startswith('qwen')
    if qwen:
        reasoning=encode('Reasoning.\n');close=encode('</think>');answer=encode('FINAL ANSWER: 42')
        ids=reasoning+close+answer+[eos[0]]
        natural=partition_generated(ids,tokenizer,model_key,len(ids)+10,eos,prompt_think_open=True)
        if not natural['thinking_boundary_found'] or natural['reasoning_right_censored']:
            raise Blocked('Synthetic closed think partition failed')
        if sum(natural[k] for k in ('reasoning_tokens','answer_tokens','structural_tokens'))!=len(ids):
            raise Blocked('Synthetic token conservation failed')
        capped=partition_generated(reasoning,tokenizer,model_key,len(reasoning),eos,prompt_think_open=True)
        if capped['final_text']!='' or not capped['reasoning_right_censored']:
            raise Blocked('Synthetic unclosed think leaks answer text')
        boundary_at_cap=partition_generated(ids,tokenizer,model_key,len(ids),eos,prompt_think_open=True)
        if boundary_at_cap['output_right_censored']:raise Blocked('EOS-at-cap incorrectly censored')
        return {'status':'PASS','synthetic_tokenizer_test':True,'cases':3,
                'think_open_ids':encode('<think>'),'think_close_ids':close,
                'closed_partition':natural,'unclosed_partition':capped,
                'eos_at_cap_not_censored':True}
    ids=encode('Reasoning.\nFINAL ANSWER: 42')+[eos[0]]
    result=partition_generated(ids,tokenizer,model_key,len(ids),eos)
    if result['marker_count']!=1 or result['explicit_explanation_token_bounds'] is None:
        raise Blocked('Synthetic final marker boundary missing')
    if result['output_right_censored'] or not result['eos_terminated']:
        raise Blocked('Synthetic EOS partition failed')
    return {'status':'PASS','synthetic_tokenizer_test':True,'cases':1,
            'marker_partition':result,
            'cross_token_observed':result['boundary_crosses_token'],
            'cross_token_note':'Real tokenizer may place the marker exactly at a token boundary; no crossing is fabricated.'}


def replay_natural(directory,tokenizer,model_key,revision,eos_token_ids):
    directory=Path(directory);identity_path=directory/'identity.json';natural_path=directory/'natural.json'
    if not identity_path.exists() or not natural_path.exists():
        return {'status':'INCONCLUSIVE','directory':str(directory),'reason':'Actual identity/natural receipt missing'}
    identity,natural=read(identity_path),read(natural_path)
    if identity.get('model_revision')!=revision:raise Blocked('Natural diagnostic model revision mismatch')
    prompt=identity.get('prompt_ids')
    if isinstance(prompt,list) and len(prompt)==1 and isinstance(prompt[0],list):prompt=prompt[0]
    if not prompt or not isinstance(prompt,list) or any(type(t) is not int for t in prompt):
        return {'status':'INCONCLUSIVE','directory':str(directory),'reason':'Actual prompt IDs absent; never guessed or retokenized'}
    record=dict(natural,uid=identity['uid'],model=model_key,record_complete=True,
                prompt_sha256=digest(json.dumps(prompt)),generation={'max_new_tokens':identity['natural_cap']})
    enriched=enrich_record(record,{'uid':identity['uid'],'model':model_key,'prompt_ids':prompt},tokenizer,eos_token_ids)
    fields=('output_tokens','reasoning_tokens','answer_tokens','structural_tokens','output_cap_reached',
            'eos_terminated','thinking_boundary_found','reasoning_right_censored','boundary_anomalies',
            'explicit_explanation_tokens','explicit_explanation_token_bounds','final_region_parse_failure','final_text_scope')
    return {'status':'PASS','directory':str(directory),'scope':'Existing single-image natural diagnostic token parsing only',
            'identity_sha256':digest(identity_path.read_bytes()),'natural_sha256':digest(natural_path.read_bytes()),
            'partition':{key:enriched.get(key) for key in fields}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--model',required=True,choices=['llava7','mini8','qwen8','qwen4'])
    p.add_argument('--natural-dirs',type=Path,nargs='*',default=[]);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise Blocked('Use a new immutable tokenizer probe path')
    lock=read(a.folder/'manifests/model_weights_lock.json')[a.model];model_path=Path(lock['path'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),use_fast=a.model.startswith('qwen'),
                                           local_files_only=True,trust_remote_code=False)
    config_path=model_path/'generation_config.json';generation=read(config_path)
    if 'eos_token_id' not in generation:raise Blocked('Actual frozen generation EOS missing')
    eos=generation['eos_token_id']
    synthetic=synthetic_checks(tokenizer,a.model,eos)
    natural=[replay_natural(d,tokenizer,a.model,lock['revision'],eos) for d in a.natural_dirs]
    names=('tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt')
    assets={name:digest((model_path/name).read_bytes()) for name in names if (model_path/name).exists()}
    report={'status':'PASS' if all(r['status']=='PASS' for r in natural) else 'INCONCLUSIVE',
            'model':a.model,'revision':lock['revision'],'native_gate_passed':False,'production_ready':False,
            'scope':'Tokenizer parsing only; no model loading, generation, or GPU execution',
            'tokenizer_class':type(tokenizer).__name__,'tokenizer_assets_sha256':assets,
            'generation_config_sha256':digest(config_path.read_bytes()),
            'synthetic':synthetic,'existing_natural':natural}
    publish_once(a.out,report)
    print(json.dumps({'status':report['status'],'out':str(a.out),'existing_natural_records':len(natural)}))


if __name__=='__main__':main()
