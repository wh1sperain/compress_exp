"""Qwen author public loader versus local setup, in separate fresh processes.

Independence is at construction/entrypoint level: both use the same pinned
author forward implementation. This does NOT independently validate the paper's
algorithm or establish the complete native/backend gate.
"""
import argparse
import copy
import time
from pathlib import Path
from .common import read, write, digest, Blocked


PROBES = (0, 1, 127, 128, 511, 512, 1023, 1024)


def tensor_record(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [tensor_record(x) for x in value]
    if isinstance(value, dict):
        return {str(k): tensor_record(v) for k, v in value.items()}
    import torch
    x = value.detach().contiguous().cpu()
    return dict(shape=list(x.shape), dtype=str(x.dtype),
                sha256=digest(x.view(torch.uint8).numpy().tobytes()))


def prior_formula(attention, image_range):
    """Independent expression of the frozen Qwen single-forward score."""
    import torch
    start, end = image_range
    q = attention[:, :, end, start:end].mean(1).squeeze(0).float()
    p = attention[:, :, end:-1, start:end].mean(2).mean(1).squeeze(0).float()
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    return p, q, p * torch.log((p + 1e-10) / (q + 1e-10))


def run(folder, candidate, method, side, out, model_key='qwen8'):
    import torch
    from PIL import Image
    from .qwen_smoke import activate, run as load_local
    from .audit_qwen_inputs import question_text
    from .correctness_probe import cache_lengths
    if out.exists():
        raise Blocked('Immutable attempt already exists')
    out.mkdir(parents=True)
    lock = read(folder / 'manifests/model_weights_lock.json')[model_key]
    row = read(candidate)
    if digest(Path(row['image_path']).read_bytes()) != row['image_sha256']:
        raise Blocked('Diagnostic image changed')
    if side == 'local':
        model, processor, _ = load_local(folder, method, out / 'loading.json', model_key, load_only=True)
    else:
        activate(folder)
        from visual_token_pruning import VTRConfig
        from visual_token_pruning.model.vtr_qwen3_vl import VTRQwen3VLForConditionalGeneration
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
        cfg = VTRConfig(enabled=method != 'Full', strategy='priortr', prune_layer=3,
                        keep_ratio={'Full': 1., 'PriorTR50': .5, 'PriorTR25': .25}[method],
                        keep_tokens=None, query_aggregation='question', head_aggregation='mean')
        # Call the frozen public entrypoint; do not reproduce its post-load setup.
        model = VTRQwen3VLForConditionalGeneration.from_pretrained_vtr(
            lock['path'], vtr_config=cfg, dtype=torch.bfloat16,
            device_map={'': 'cuda:0'}, attn_implementation='eager', local_files_only=True)
        processor = Qwen3VLProcessor.from_pretrained(lock['path'], local_files_only=True,
                                                    min_pixels=262144, max_pixels=4194304)
        model.eval().requires_grad_(False)
    from transformers import LogitsProcessorList
    spec = read(Path(__file__).resolve().parent.parent / 'round3_spec.json')
    with Image.open(row['image_path']) as source:
        image = source.convert('RGB')
    question = question_text(row, spec['prompt']['suffix'])
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if inputs['input_ids'].shape[1] > 8192:
        raise Blocked('Frozen prompt cap exceeded')
    neutral = processor.tokenizer.encode(' The image contains information that can be examined carefully.', add_special_tokens=False)
    forced = (neutral * (1025 // len(neutral) + 1))[:1025]
    identity = dict(method=method, side=side, model_key=model_key, model_revision=lock['revision'],
                    author_revision=read(folder / 'manifests/source_lock.json')['PriorTR']['revision'],
                    uid=row['uid'], image_sha256=row['image_sha256'], inputs=tensor_record(inputs),
                    forced_sha256=digest(str(forced)), steps=1025)
    write(out / 'identity.json', identity)
    traces, kv, selectors, deepstack, logits, mrope = [], [], [], [], {}, []
    step = [0]
    handles, restore = [], []
    def replace(obj, name, function):
        restore.append((obj, name, getattr(obj, name)))
        setattr(obj, name, function)
    def pre(layer):
        def hook(module, args, kwargs):
            if step[0] in PROBES:
                traces.append(dict(step=step[0], layer=layer,
                    inputs={k: tensor_record(kwargs.get(k)) for k in
                            ('position_ids', 'cache_position', 'attention_mask', 'position_embeddings')}))
        return hook
    for i, layer in enumerate(model.model.language_model.layers):
        handles.append(layer.register_forward_pre_hook(pre(i), with_kwargs=True))
    def rope_hook(module, args, kwargs):
        if step[0] in PROBES:
            positions = args[1] if len(args) > 1 else kwargs.get('position_ids')
            mrope.append(dict(step=step[0], positions=tensor_record(positions)))
    handles.append(model.model.language_model.rotary_emb.register_forward_pre_hook(rope_hook, with_kwargs=True))
    def post(module, args, output):
        kv.append(cache_lengths(output.past_key_values))
    handles.append(model.register_forward_hook(post))
    if method != 'Full':
        strategy = model.vtr_strategy
        score_original, select_original = strategy.compute_scores, strategy.select_tokens
        def score(attention, image_token_range, config, *args, **kwargs):
            value = score_original(attention, image_token_range, config, *args, **kwargs)
            p, q, expected = prior_formula(attention, image_token_range)
            if not torch.equal(value, expected.to(value.dtype)):
                raise Blocked('Q/P formula mismatch')
            selectors.append(dict(kind='scores', image_range=list(image_token_range),
                                  P=tensor_record(p), Q=tensor_record(q), scores=tensor_record(value)))
            return value
        def select(scores, num_tokens, config, *args, **kwargs):
            value = select_original(scores, num_tokens, config, *args, **kwargs)
            expected = scores.topk(max(1, int(num_tokens * config.keep_ratio))).indices.sort().values
            if not torch.equal(value, expected):
                raise Blocked('Actual TopK mismatch')
            selectors.append(dict(kind='actual_selection', indices=value.cpu().tolist()))
            return value
        replace(strategy, 'compute_scores', score)
        replace(strategy, 'select_tokens', select)
        from visual_token_pruning.model.deepstack_handler import DeepStackSyncHandler
        original_ds = DeepStackSyncHandler.prune_deepstack_features
        def prune_ds(self, deepstack_visual_embeds, keep_img_indices, current_layer_idx):
            values = original_ds(self, deepstack_visual_embeds, keep_img_indices, current_layer_idx)
            for i, (before, after) in enumerate(zip(deepstack_visual_embeds, values)):
                expected = before
                if before is not None and before.numel() and i >= current_layer_idx:
                    expected = before.index_select(0, keep_img_indices.to(before.device))
                if expected is None:
                    if after is not None: raise Blocked('DeepStack None mismatch')
                elif not torch.equal(expected, after):
                    raise Blocked('DeepStack index selection mismatch')
            deepstack.append(dict(layer=current_layer_idx, before=tensor_record(deepstack_visual_embeds),
                                  after=tensor_record(values), indices=keep_img_indices.cpu().tolist()))
            return values
        replace(DeepStackSyncHandler, 'prune_deepstack_features', prune_ds)
        original_mask = DeepStackSyncHandler.update_visual_pos_masks
        def update_mask(self, visual_pos_masks, keep_indices):
            value = original_mask(self, visual_pos_masks, keep_indices)
            if not torch.equal(value, visual_pos_masks.index_select(1, keep_indices)):
                raise Blocked('DeepStack visual mask mismatch')
            deepstack.append(dict(kind='visual_pos_masks', before=tensor_record(visual_pos_masks),
                                  after=tensor_record(value), indices=keep_indices.cpu().tolist()))
            return value
        replace(DeepStackSyncHandler, 'update_visual_pos_masks', update_mask)
    class Force:
        def __call__(self, ids, scores):
            i = step[0]
            if i in PROBES: logits[i] = scores[0].detach().float().cpu()
            result = torch.full_like(scores, -float('inf'))
            result[:, forced[i]] = 0
            step[0] += 1
            if step[0] % 128 == 0:
                write(out / 'status.json', dict(stage='teacher_forcing', steps=step[0], updated_at=time.time()))
            return result
    try:
        generation = copy.deepcopy(model.generation_config)
        generation.eos_token_id = None
        generation.forced_eos_token_id = None
        # EOS disabled solely for a fixed teacher-forcing diagnostic, never research.
        with torch.inference_mode():
            model.generate(**inputs, generation_config=generation, max_new_tokens=1025,
                           do_sample=False, use_cache=True, logits_processor=LogitsProcessorList([Force()]))
        if step[0] != 1025 or len(kv) != 1025: raise Blocked('Incomplete trajectory')
        if any(len(x) != 36 for x in kv): raise Blocked('Expected all 36 cache layers')
        if any(any(b != a + 1 for a, b in zip(x, y)) for x, y in zip(kv, kv[1:])):
            raise Blocked('Cache growth mismatch')
        if method != 'Full' and (len(selectors) != 2 or not deepstack):
            raise Blocked('Pruning/DeepStack not observed')
        write(out / 'status.json', dict(status='PASS', scope='single entrypoint trajectory only', native_gate_passed=False))
    except BaseException as exc:
        write(out / 'status.json', dict(status='FAIL', error=str(exc)[:1200], native_gate_passed=False))
        raise
    finally:
        for handle in handles: handle.remove()
        for obj, attr, value in reversed(restore): setattr(obj, attr, value)
        write(out / 'trace.json', dict(layers=traces, kv=kv, selectors=selectors, deepstack=deepstack, mrope=mrope))
        torch.save(logits, out / 'logits.pt')


def check_identity(a, b):
    if {a['side'], b['side']} != {'author', 'local'}:
        raise Blocked('Comparison must use distinct author and local entrypoints')
    for key in ('method', 'model_key', 'model_revision', 'author_revision', 'uid', 'image_sha256', 'inputs', 'forced_sha256', 'steps'):
        if a[key] != b[key]: raise Blocked('Different comparison identity: ' + key)


def compare(author, local):
    import torch
    a, b = read(author / 'identity.json'), read(local / 'identity.json')
    check_identity(a, b)
    for folder in (author, local):
        if read(folder / 'status.json')['status'] != 'PASS': raise Blocked('Incomplete or failed worker')
    x = torch.load(author / 'logits.pt', map_location='cpu', weights_only=True)
    y = torch.load(local / 'logits.pt', map_location='cpu', weights_only=True)
    if set(x) != set(PROBES) or set(y) != set(PROBES): raise Blocked('Missing numerical probe')
    metrics = []
    for i in PROBES:
        u, v = x[i].double(), y[i].double()
        if not torch.isfinite(u).all() or not torch.isfinite(v).all(): raise Blocked('Nonfinite logits')
        p, q = u.log_softmax(-1), v.log_softmax(-1)
        metrics.append(dict(step=i, kl=float((p.exp() * (p-q)).sum()),
                            tv=float((p.exp()-q.exp()).abs().sum()/2), max_abs=float((u-v).abs().max())))
    exact_trace = read(author / 'trace.json') == read(local / 'trace.json')
    passed = exact_trace and all(m['kl'] < .02 and m['tv'] < .08 for m in metrics)
    return dict(status='PASS' if passed else 'FAIL', native_gate_passed=False,
                scope='independent public loader vs local construction, shared pinned author forward; one image, eight numerical probes, no natural/backend certification',
                exact_positions_masks_kv_selectors_deepstack=exact_trace, metrics=metrics, identity=a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder'); parser.add_argument('--candidate')
    parser.add_argument('--method', choices=['Full', 'PriorTR50', 'PriorTR25'])
    parser.add_argument('--side', choices=['author', 'local'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--compare', nargs=2, metavar=('AUTHOR', 'LOCAL'))
    args = parser.parse_args()
    if Path(args.out).exists(): raise Blocked('Refuse overwrite')
    if args.compare:
        write(args.out, compare(*(Path(x) for x in args.compare)))
    else:
        if not all((args.folder, args.candidate, args.method, args.side)): parser.error('Worker arguments required')
        run(Path(args.folder), Path(args.candidate), args.method, args.side, Path(args.out))
