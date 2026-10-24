"""R3 FastV curve and random-mask controls on the frozen static author path.

Random25_L2 replaces only the author's visual top-k selection expression.
All text retention, pruning layer, positions, mask construction, and per-layer
cache handling remain the author's code. This is an explicit random control,
not an additional claimed FastV reproduction. GPU certification is separate.
"""
from __future__ import annotations
import ast
import copy
import functools
import inspect
import json
import types
from .common import Blocked, digest
from .native_sources import AuthorSources

FASTV_RELATIVE = 'src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py'
FASTV_KEEP = {'FastV75': 432, 'FastV50': 288, 'FastV37p5': 216, 'FastV25': 144}


def uses_fastv_runtime(condition):
    return condition in FASTV_KEEP or condition in ('FastVFull', 'Random25_L2')


def random_mask(uid):
    """Frozen PCG64 permutation, independent of repeat and generation RNG."""
    import numpy as np
    if not isinstance(uid, str) or not uid:
        raise Blocked('Random mask needs a nonempty immutable question uid')
    seed = int(digest('r3-mask|' + uid)[:16], 16)
    indices = sorted(np.random.Generator(np.random.PCG64(seed)).permutation(576)[:144].tolist())
    return {'mask_seed': seed, 'visual_indices': indices,
            'mask_algorithm': 'numpy.PCG64.permutation(576)[:144];sorted',
            'mask_sha256': digest(json.dumps(indices))}


def random_forward_ast(source):
    """Fail closed if the pinned author's two selection sites have changed."""
    tree = ast.parse(source)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LlamaModel'), None)
    if cls is None:
        raise Blocked('Frozen FastV LlamaModel missing')
    method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'fastv_forward'), None)
    if method is None:
        raise Blocked('Frozen FastV forward missing')
    method = copy.deepcopy(method)
    expected = ast.parse('last_layer_attention_avg_last_tok_image.topk(round(FASTV_image_token_length*(1-FASTV_r))).indices + FASTV_image_token_start_index', mode='eval').body
    replacements = 0
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'top_attention_rank_index' for t in node.targets):
            if ast.dump(node.value) != ast.dump(expected):
                raise Blocked('FastV visual selector no longer matches pinned expression')
            node.value = ast.parse('_r3_random_indices(device, FASTV_image_token_start_index, FASTV_image_token_length, FASTV_r)', mode='eval').body
            replacements += 1
    if replacements != 2:
        raise Blocked('Expected exactly cached and uncached author selection sites')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    return ast.fix_missing_locations(module)


def configure_fastv_control(model, condition, image_start, *, uid=None, source_lock=None):
    """Configure one B1 request; call after its actual prompt is tokenized.

    Caller must use the native-fastv environment/compatibility forward. Returned
    evidence should be persisted with the request. No checkpoint file is edited.
    """
    if not isinstance(image_start, int) or image_start < 0:
        raise Blocked('Invalid image start')
    if condition not in FASTV_KEEP and condition != 'Random25_L2':
        raise Blocked('Unregistered FastV control: ' + condition)
    decoder = model.model
    original = decoder.fastv_forward
    keep = 144 if condition == 'Random25_L2' else FASTV_KEEP[condition]
    config = dict(fastv_k=2, fastv_r=1-keep/576,
                  image_token_start_index=image_start, image_token_length=576)
    metadata = {'condition': condition, 'target_visual_tokens': keep,
                'prune_after_layers': 2, 'cache_policy': 'frozen-FastV-static',
                'image_token_start_index': image_start}
    function = original
    if condition == 'Random25_L2':
        if source_lock is None:
            raise Blocked('Random control needs the frozen author source lock')
        source_path = AuthorSources(source_lock).file('FastV', FASTV_RELATIVE)
        # Bind to the same verified implementation already imported by the model.
        if source_path.resolve() != __import__('pathlib').Path(original.__func__.__code__.co_filename).resolve():
            raise Blocked('Loaded FastV function is not the locked author file')
        mask = random_mask(uid)
        metadata.update(mask)
        module = random_forward_ast(source_path.read_text(encoding='utf-8-sig'))
        def selected(device, start, length, ratio):
            import torch
            if length != 576 or round(length*(1-ratio)) != 144:
                raise Blocked('Random25 budget changed')
            return torch.tensor(mask['visual_indices'], device=device, dtype=torch.long) + start
        scope = dict(original.__func__.__globals__)
        scope['_r3_random_indices'] = selected
        exec(compile(module, str(source_path) + ':R3-random-control', 'exec'), scope)
        function = types.MethodType(scope['fastv_forward'], decoder)
        metadata.update(author_file_sha256=digest(source_path.read_bytes()),
                        transformed_ast_sha256=digest(ast.dump(module)))
    configured = functools.partial(function, fastv_config=config)
    signature = inspect.signature(original)
    @functools.wraps(original)
    def guarded(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        if 'fastv_config' in bound.arguments:
            raise Blocked('Do not override registered pruning configuration')
        cache = bound.arguments.get('use_cache')
        if cache is None:
            cache = decoder.config.use_cache
        if not cache:
            raise Blocked('R3 FastV/random controls require static cached decoding')
        for name in ('input_ids', 'inputs_embeds'):
            tensor = bound.arguments.get(name)
            if tensor is not None and tensor.shape[0] != 1:
                raise Blocked('This author control adapter is B1 only')
        return configured(*args, **kwargs)
    decoder.forward = guarded
    return metadata
