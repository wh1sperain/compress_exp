"""Mini native prefill plus actual multi-request HF Llama decode.

One resident model; each prompt keeps its original visual compression and four
prefusion layers. Decode uses the installed native Llama forward (including its
Llama3.1 RoPE implementation), explicit per-row positions and masked left-padded
KV. This is a padded HF batch backend, not a paged/Graph backend or a certified
equivalence claim. The scheduler alone owns RNG, sampling, stopping and storage.
"""
from .common import Blocked


def legacy_cache(cache):
    if hasattr(cache, 'to_legacy_cache'):
        cache = cache.to_legacy_cache()
    if not isinstance(cache, (tuple, list)) or not cache:
        raise Blocked('Mini requires nonempty native KV cache')
    result = tuple(tuple(layer) for layer in cache)
    for layer in result:
        if len(layer) != 2 or layer[0].ndim != 4 or layer[1].shape != layer[0].shape or layer[0].shape[0] != 1:
            raise Blocked('Unexpected single-request Mini KV layout')
    if len({int(layer[0].shape[-2]) for layer in result}) != 1:
        raise Blocked('Mini layers must retain equal native sequence lengths')
    return result


def layout(lengths):
    """Public CPU-only description of the actual left-padding mask/positions."""
    if not lengths or any(type(n) is not int or n <= 0 for n in lengths):
        raise Blocked('Expected positive per-request cache lengths')
    maximum = max(lengths)
    return dict(maximum=maximum, left_padding=[maximum-n for n in lengths],
                position_ids=[[n] for n in lengths],
                attention_mask=[[0]*(maximum-n)+[1]*(n+1) for n in lengths])


def pack(states):
    import torch
    if len({id(state) for state in states}) != len(states):
        raise Blocked('A request cannot occupy two batch slots')
    caches = [legacy_cache(state['opaque']['cache']) for state in states]
    if len({len(cache) for cache in caches}) != 1:
        raise Blocked('Mixed Mini layer counts')
    lengths = [int(cache[0][0].shape[-2]) for cache in caches]
    if any(state['opaque']['next_position'] != n for state, n in zip(states, lengths)):
        raise Blocked('Mini position/cache length mismatch')
    description = layout(lengths)
    packed = []
    for layer in range(len(caches[0])):
        pair = []
        for kind in range(2):
            values = [cache[layer][kind] for cache in caches]
            prototype = values[0]
            if any(value.shape[1] != prototype.shape[1] or value.shape[3] != prototype.shape[3]
                   or value.dtype != prototype.dtype or value.device != prototype.device for value in values):
                raise Blocked('Mixed Mini KV geometry/device/dtype')
            pair.append(torch.cat([torch.nn.functional.pad(value, (0, 0, pad, 0))
                                   for value, pad in zip(values, description['left_padding'])], dim=0))
        packed.append(tuple(pair))
    device = caches[0][0][0].device
    return tuple(packed), torch.tensor(description['attention_mask'], device=device), torch.tensor(
        description['position_ids'], device=device), lengths


class Backend:
    name = 'mini-native-prefill-hf-padded-batch'

    def __init__(self, session):
        self.session = session
        self.model = session.model
        if session.key != 'mini8' or session.condition != 'MiniNative':
            raise Blocked('Mini backend requires mini8/MiniNative')
        if self.model.config.compressor_size != 1 or len(self.model.model.prefusion_layers) != 4:
            raise Blocked('Mini trained one-token/four-prefusion configuration required')
        # Resolve the native Llama implementation, preserving the environment's
        # checkpoint-specific rotary scaling. Never reproduce old LLaVA RoPE.
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        if not isinstance(self.model, LlamaForCausalLM):
            raise Blocked('Unexpected Mini causal-LM base class')
        self.native_forward = LlamaForCausalLM.forward

    def prefill(self, row, job):
        import copy
        import torch
        inputs, metadata = self.session.prepare(row)
        outputs = []
        evidence = {'prefusion_lengths': {}}
        def capture(module, args, output):
            outputs.append((output.logits[:, -1, :].detach(), legacy_cache(output.past_key_values)))
        def compression(module, args, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            evidence['actual_visual_tokens'] = int(tensor.shape[1])
        def prefusion(index):
            def record(module, args, kwargs):
                hidden = args[0] if args else kwargs['hidden_states']
                evidence['prefusion_lengths'][str(index)] = int(hidden.shape[1])
            return record
        hooks = [self.model.register_forward_hook(capture),
                 self.model.model.compressor.register_forward_hook(compression)]
        hooks += [layer.register_forward_pre_hook(prefusion(i), with_kwargs=True)
                  for i, layer in enumerate(self.model.model.prefusion_layers)]
        config = copy.deepcopy(self.model.generation_config)
        config.max_new_tokens = 1
        config.do_sample = False
        config.num_beams = 1
        config.use_cache = True
        # The single greedy dummy token is discarded; no research RNG advances.
        try:
            with torch.inference_mode():
                self.model.generate(**inputs, generation_config=config)
        finally:
            for hook in hooks:
                hook.remove()
        if len(outputs) != 1:
            raise Blocked('Mini prefill must execute exactly one full prompt forward')
        logits, cache = outputs[0]
        lengths = [int(pair[0].shape[-2]) for pair in cache]
        audit = row['native_inputs']['mini8']
        if evidence.get('actual_visual_tokens') != 1 or evidence['prefusion_lengths'] != audit['prefusion_lengths']:
            raise Blocked('Mini actual compression/prefusion differs from frozen audit')
        if any(n != audit['expanded_prompt_tokens'] for n in lengths):
            raise Blocked('Mini prefill KV differs from frozen audit')
        metadata = dict(metadata, **evidence)
        return dict(logits=logits, prefill_layer_kv=lengths, metadata=metadata,
                    opaque=dict(cache=cache, next_position=lengths[0]))

    def decode(self, states, tokens_tensor):
        import torch
        if not states or tokens_tensor.ndim != 1 or tokens_tensor.shape[0] != len(states):
            raise Blocked('One sampled token required per live request')
        packed, mask, positions, lengths = pack(states)
        with torch.inference_mode():
            # Direct base forward avoids re-running multimodal preprocessing.
            output = self.native_forward(self.model, input_ids=tokens_tensor[:, None],
                past_key_values=packed, attention_mask=mask, position_ids=positions,
                use_cache=True, return_dict=True)
        cache = output.past_key_values
        if hasattr(cache, 'to_legacy_cache'):
            cache = cache.to_legacy_cache()
        if len(cache) != len(packed) or any(pair[0].shape[-2] != max(lengths)+1 for pair in cache):
            raise Blocked('Mini batched KV failed to grow by exactly one')
        # Clone trimmed rows: finished slots can release memory immediately,
        # and no request retains another request's padded backing storage.
        for index, (state, length) in enumerate(zip(states, lengths)):
            state['opaque'] = dict(cache=tuple(tuple(value[index:index+1, :, -(length+1):, :].clone()
                for value in pair) for pair in cache), next_position=length+1)
            state['logits'] = output.logits[index:index+1, -1, :].detach()
        return output.logits[:, -1, :]

    def kv_lengths(self, state):
        return [int(pair[0].shape[-2]) for pair in state['opaque']['cache']]

    def kv_bytes(self, state):
        return sum(value.numel()*value.element_size() for pair in state['opaque']['cache'] for value in pair)

    def release(self, state):
        state['opaque'] = None
        state['logits'] = None


MiniBatchBackend = Backend
