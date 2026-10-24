"""Independent Mini builder integration diagnostic; never a complete native gate.

Two separate processes load checkpoint via session / frozen author builder. Both
use the same fixed author model class, so this tests integration, NOT an
independently reimplemented compression algorithm. The builder's FP16 constants
are explicitly changed to BF16; checkpoint vision is materialized before loading.
"""
import argparse
import ast
import json
from pathlib import Path
from .common import Blocked, digest, read, write


def bf16_builder_tree(source):
    tree = ast.parse(source)
    class Convert(ast.NodeTransformer):
        count = 0
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == 'torch' and node.attr == 'float16':
                node.attr = 'bfloat16'; self.count += 1
            return node
    conversion = Convert(); tree = conversion.visit(tree)
    if conversion.count != 5:
        raise Blocked('Frozen author builder dtype sites changed: ' + str(conversion.count))
    return ast.fix_missing_locations(tree), conversion.count


def require_strict(info):
    for name in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if info.get(name):
            raise Blocked('Author checkpoint loading mismatch: ' + name)


def verify_vision_checkpoint(model, model_path):
    """Compare loaded vision tensors to checkpoint, never to external CLIP."""
    import torch
    model_path = Path(model_path)
    index_path = next((model_path/n for n in ('model.safetensors.index.json','pytorch_model.bin.index.json') if (model_path/n).exists()), None)
    if index_path is None: raise Blocked('Sharded checkpoint index required for vision proof')
    mapping = read(index_path)['weight_map']
    names = [k for k in mapping if k.startswith('model.vision_tower.')]
    if not names: raise Blocked('Checkpoint contains no vision tensors')
    state = model.state_dict(); checked = 0
    for shard in sorted({mapping[k] for k in names}):
        keys = [k for k in names if mapping[k] == shard]
        path = model_path/shard
        if path.suffix == '.safetensors':
            from safetensors import safe_open
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                for key in keys:
                    expected = handle.get_tensor(key).to(torch.bfloat16)
                    if key not in state or not torch.equal(state[key].detach().cpu(),expected):
                        raise Blocked('Checkpoint vision tensor mismatch: '+key)
                    checked += 1
        else:
            tensors = torch.load(path,map_location='cpu',weights_only=True)
            for key in keys:
                if key not in state or not torch.equal(state[key].detach().cpu(),tensors[key].to(torch.bfloat16)):
                    raise Blocked('Checkpoint vision tensor mismatch: '+key)
                checked += 1
            del tensors
    return {'verified_tensors':checked,'checkpoint_index_sha256':digest(index_path.read_bytes())}


def finalize_bf16(model, device):
    """Explicit study-wide conversion after strict checkpoint dispatch.

Reject unmaterialized tensors before .to(); do not replace/reset any module or
reload external CLIP. HF can retain FP32 submodules despite torch_dtype.
"""
    import torch
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    meta = [name for name, tensor in tensors if tensor.is_meta]
    if meta: raise Blocked('Unmaterialized checkpoint tensors: '+', '.join(meta[:12]))
    before = {name: str(tensor.dtype) for name,tensor in model.named_parameters()
              if tensor.is_floating_point() and tensor.dtype != torch.bfloat16}
    model.to(device=device, dtype=torch.bfloat16)
    remaining = [name for name,p in model.named_parameters()
                 if p.is_meta or (p.is_floating_point() and p.dtype != torch.bfloat16)]
    if remaining: raise Blocked('BF16 conversion incomplete: '+', '.join(remaining[:12]))
    return {'non_bf16_parameters_before_conversion':before,
            'all_parameters_materialized':True,'all_floating_parameters_bf16':True}


def author_load(folder, out):
    import importlib
    import torch
    from .native_smoke import activate
    paths = activate(folder, 'MiniNative')
    builder = importlib.import_module('llavamini.model.builder')
    language = importlib.import_module('llavamini.model.language_model.llavamini_llama')
    cls = language.LlavaMiniLlamaForCausalLM
    cfgcls = language.LlavaMiniConfig
    lock = read(folder / 'manifests/model_weights_lock.json')
    model_path = lock['mini8']['path']
    cfg = cfgcls.from_pretrained(model_path, local_files_only=True)
    cfg.mm_vision_tower = lock['clip336']['path']
    cfg.unfreeze_mm_vision_tower = True  # Constructor only; freeze every parameter below.
    source_path = paths['LLaVA-Mini'] / 'llavamini/model/builder.py'
    source = source_path.read_text(encoding='utf-8')
    tree, count = bf16_builder_tree(source)
    ns = dict(vars(builder))
    loading = {}
    class StrictCheckpoint:
        @staticmethod
        def from_pretrained(path, **kwargs):
            if str(path) != str(model_path): raise Blocked('Unexpected model base')
            kwargs.update(config=cfg, output_loading_info=True, local_files_only=True,
                          attn_implementation='eager')
            model, info = cls.from_pretrained(path, **kwargs)
            write(out / 'author_loading_info.json', info)
            require_strict(info); loading.update(info)
            if not model.get_vision_tower().is_loaded:
                raise Blocked('Checkpoint vision was not constructed before loading')
            return model
    ns['LlavaMiniLlamaForCausalLM'] = StrictCheckpoint
    exec(compile(tree, str(source_path), 'exec'), ns)
    # exec's wildcard import restores the original name; bind the loading guard after it.
    ns['LlavaMiniLlamaForCausalLM'] = StrictCheckpoint
    tokenizer, model, processor, context = ns['load_pretrained_model'](
        model_path, None, 'llava-mini-llama-3.1-8b', device_map='cuda:0', device='cuda',
        local_files_only=True, attn_implementation='eager')
    model.eval().requires_grad_(False)
    conversion = finalize_bf16(model, 'cuda:0')
    write(out/'author_dtype_conversion.json',conversion)
    return model, tokenizer, processor, dict(
        source_sha256=digest(source), dtype_ast_sites=count, loading_info=loading,
        dtype_conversion=conversion,
        loader_adjustments=['FP16 constants to BF16', 'local pinned CLIP path',
            'eager checkpoint vision construction before strict dispatch', 'eval and frozen parameters',
            'explicit full-model BF16 conversion after strict dispatch; reject meta tensors'],
        author_context_return=context)


def author_prepare(model, tokenizer, processor, row):
    import torch
    from PIL import Image
    from llavamini.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llavamini.conversation import conv_templates
    from llavamini.mm_utils import process_images, tokenizer_image_token
    if digest(Path(row['image_path']).read_bytes()) != row['image_sha256']:
        raise Blocked('Image changed')
    spec = read(Path(__file__).resolve().parent.parent / 'round3_spec.json')
    # Use the shared study question text, but the author conversation/image helpers directly.
    # Template/processor/model integration is independent of NativeSession.prepare.
    from .audit_qwen_inputs import question_text
    conversation = conv_templates['llava_llama_3_1'].copy()
    conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN+'\n'+question_text(row,spec['prompt']['suffix']))
    conversation.append_message(conversation.roles[1], None)
    ids = tokenizer_image_token(conversation.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    with Image.open(row['image_path']) as image:
        pixels = process_images([image.convert('RGB')], processor, model.config)
    if model.config.resolution_ratio != 1 or not isinstance(pixels, torch.Tensor):
        raise Blocked('Unsupported Mini image path')
    return {'inputs': ids.to('cuda:0'), 'images': pixels.unsqueeze(0).to(device='cuda:0',dtype=torch.bfloat16)}


def run(folder, mode, manifest, uid, out):
    import torch
    from transformers import LogitsProcessorList
    from .correctness_probe import cache_lengths
    if out.exists(): raise Blocked('Do not overwrite diagnostic attempt')
    out.mkdir(parents=True)
    if manifest.name != 'calibration_manifest.jsonl': raise Blocked('Only frozen calibration inputs allowed')
    rows = [json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines()]
    row = next((r for r in rows if r['uid'] == uid), None)
    if row is None: raise Blocked('UID must be in frozen calibration manifest')
    if mode == 'author':
        model, tokenizer, processor, loading = author_load(folder, out)
        inputs = author_prepare(model, tokenizer, processor, row)
    else:
        from .native_session import NativeSession
        session = NativeSession(folder, 'mini8', 'MiniNative', out/'session-load.json')
        model, tokenizer = session.model, session.tokenizer
        require_strict(session.loading)
        inputs, metadata = session.prepare(row)
        loading = dict(session_loading=session.loading, prepare_metadata=metadata)
    lock = read(folder/'manifests/model_weights_lock.json')
    loading['vision_checkpoint_proof'] = verify_vision_checkpoint(model,lock['mini8']['path'])
    if model.config.compressor_size != 1 or len(model.model.prefusion_layers) != 4:
        raise Blocked('Expected trained 1-token, 4-prefusion configuration')
    blobs = {}; handles = []; kv = []; steps = [0]; snapshots = {}
    def save(name):
        def hook(module, args, output):
            tensor = output[0] if isinstance(output,(tuple,list)) else output
            if name not in blobs: blobs[name] = tensor.detach().cpu()
        return hook
    handles.append(model.model.compressor.register_forward_hook(save('compressor')))
    for i, layer in enumerate(model.model.prefusion_layers):
        handles.append(layer.register_forward_hook(save('prefusion_'+str(i))))
    def capture(module,args,output): kv.append(cache_lengths(output.past_key_values))
    handles.append(model.register_forward_hook(capture))
    text_ids = tokenizer.encode(' The image contains information that can be examined carefully.', add_special_tokens=False)
    forced = (text_ids*(1025//len(text_ids)+1))[:1025]
    class Force:
        def __call__(self, ids, scores):
            index = steps[0]
            if index in {0,1,127,128,511,512,1023,1024}: snapshots[index] = scores[0].detach().float().cpu()
            result = torch.full_like(scores, -float('inf')); result[:,forced[index]]=0
            steps[0]+=1
            if steps[0]%128==0: write(out/'status.json',dict(stage='forcing',completed=steps[0]))
            return result
    # Explicit copy prevents legacy EOS fallback; teacher forcing is diagnostic only.
    import copy
    config = copy.deepcopy(model.generation_config)
    config.eos_token_id = None; config.forced_eos_token_id = None
    config.max_new_tokens = 1025; config.do_sample = False; config.use_cache = True
    try:
        with torch.inference_mode():
            model.generate(**inputs,generation_config=config,logits_processor=LogitsProcessorList([Force()]))
    finally:
        for handle in handles: handle.remove()
    if steps[0] != 1025 or len(kv)!=1025: raise Blocked('Incomplete forcing')
    if blobs['compressor'].shape[1]!=1 or len(blobs)!=5: raise Blocked('Feature evidence incomplete')
    torch.save(dict(features=blobs,logits=snapshots),out/'evidence.pt')
    write(out/'receipt.json',dict(mode=mode,uid=uid,image_sha256=row['image_sha256'],
        manifest_sha256=digest(manifest.read_bytes()), prompt_ids=inputs['inputs'].cpu().tolist(),
        teacher_prefix=forced, kv=kv, loading=loading, steps=steps[0],
        scope='separate-loader integration on fixed author class; not independent algorithm implementation',
        native_gate_passed=False, status='COLLECTED'))


def compare(left, right, out):
    import torch
    if out.exists(): raise Blocked('Comparison receipt already exists')
    a=read(left/'receipt.json');b=read(right/'receipt.json')
    if a['mode'] == b['mode']: raise Blocked('Require author and session independently collected')
    for key in ('uid','image_sha256','manifest_sha256','prompt_ids','teacher_prefix','kv','steps'):
        if a[key]!=b[key]: raise Blocked('Integration identity mismatch: '+key)
    x=torch.load(left/'evidence.pt',map_location='cpu',weights_only=True)
    y=torch.load(right/'evidence.pt',map_location='cpu',weights_only=True)
    if set(x['features']) != set(y['features']) or set(x['logits']) != set(y['logits']):
        raise Blocked('Evidence keys differ')
    if len(x['features']) != 5 or len(x['logits']) != 8:
        raise Blocked('Incomplete evidence')
    features={}
    for key in x['features']:
        p=x['features'][key].float();q=y['features'][key].float()
        features[key]=bool(torch.isfinite(p).all() and torch.isfinite(q).all() and torch.equal(p,q))
    metrics={}
    for step in x['logits']:
        p=x['logits'][step].float();q=y['logits'][step].float()
        lp=p.log_softmax(-1);lq=q.log_softmax(-1)
        kl=float((lp.exp()*(lp-lq)).sum());tv=float((lp.exp()-lq.exp()).abs().sum()/2)
        metrics[str(step)]={'kl':kl,'tv':tv,'max_abs':float((p-q).abs().max())}
        if not torch.isfinite(p).all() or not torch.isfinite(q).all() or kl>=.02 or tv>=.08:
            raise Blocked('Logit numerical gate failed')
    if not all(features.values()): raise Blocked('Compressor/prefusion features differ')
    write(out,dict(status='PASS',scope=a['scope'],feature_exact=features,logits=metrics,
        steps=a['steps'],kv_exact=True,native_gate_passed=False,
        limitations=['single calibration image','8 logits positions','shared fixed author model class',
                     'builder loading adjustments explicit; original unmodified FP16 loader not certified']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path);p.add_argument('--mode',choices=['author','session','compare'],required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--uid');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--left',type=Path);p.add_argument('--right',type=Path);a=p.parse_args()
    try:
        if a.mode=='compare': compare(a.left,a.right,a.out)
        else: run(a.folder,a.mode,a.manifest,a.uid,a.out)
    except BaseException as exc:
        failure = a.out.with_name(a.out.name+'.failure.json')
        if not failure.exists():
            write(failure,dict(status='FAIL',mode=a.mode,error_type=type(exc).__name__,
                error=str(exc)[:2000],native_gate_passed=False))
        raise

if __name__=='__main__':main()
