"""Length partitions on original generated IDs, without answer retokenization.

Total output includes every generated structural token. Capped observations
are never labelled natural completion lengths. A/C explicit-explanation
boundaries inside a token are reported as intervals, not invented exact counts.
"""
from __future__ import annotations
from .common import Blocked
from .scoring import MARKER


def _matches(ids, sequence):
    if not sequence:
        raise Blocked('Empty boundary-token sequence')
    return [i for i in range(len(ids)-len(sequence)+1) if ids[i:i+len(sequence)] == sequence]


def _decode(tokenizer, ids, *, visible=False):
    return tokenizer.decode(ids, skip_special_tokens=visible, clean_up_tokenization_spaces=False)


def _think_sequences(tokenizer):
    opening = list(tokenizer.encode('<think>', add_special_tokens=False))
    closing = list(tokenizer.encode('</think>', add_special_tokens=False))
    if not opening or not closing or opening == closing:
        raise Blocked('Tokenizer does not distinguish think boundaries')
    if _decode(tokenizer, opening) != '<think>' or _decode(tokenizer, closing) != '</think>':
        raise Blocked('Think-token boundaries do not round-trip')
    return opening, closing


def prompt_has_open_think(prompt_ids, tokenizer):
    """Inspect actual prompt IDs; prompt structural tokens never count as output."""
    opening, closing = _think_sequences(tokenizer)
    opens, closes = _matches(list(prompt_ids), opening), _matches(list(prompt_ids), closing)
    return bool(opens and (not closes or opens[-1] > closes[-1]))


def explanation_boundary(ids, tokenizer):
    # Removing structural token display strings is decoding, not retokenization.
    # This prevents a terminal </s> from becoming part of the final answer line.
    text = _decode(tokenizer, ids, visible=True)
    matches = list(MARKER.finditer(text))
    result = {'marker_count': len(matches), 'first_marker_char': matches[0].start() if matches else None,
              'last_marker_char': matches[-1].start() if matches else None,
              'explicit_explanation_tokens': None, 'explicit_explanation_token_bounds': None,
              'boundary_crosses_token': False, 'boundary_source': None,
              'marker_followed_by_nonempty_text': bool(matches and text[matches[-1].end():].strip()),
              'explanation_parse_failure': not bool(matches),
              'boundary_text_view': 'decode_original_ids_skip_special_tokens_no_cleanup'}
    if not matches:
        return result
    boundary = matches[-1].start()
    lower = 0
    for k in range(len(ids)+1):
        prefix = _decode(tokenizer, ids[:k], visible=True)
        # Byte fallback decoders may temporarily emit U+FFFD. Such a prefix
        # cannot establish a character boundary; only stable prefixes can.
        if not text.startswith(prefix):
            continue
        if len(prefix) < boundary:
            lower = k
            continue
        exact = len(prefix) == boundary
        result.update(explicit_explanation_tokens=k if exact else None,
                      explicit_explanation_token_bounds=[k, k] if exact else [lower, k],
                      boundary_crosses_token=not exact,
                      boundary_source='original_ids_incremental_decode')
        return result
    raise Blocked('Original-token boundary could not be mapped')


def partition_generated(ids, tokenizer, model_key, max_new_tokens, eos_token_ids,
                        *, prompt_think_open=None):
    """Return generation-only metadata; does not score or change any output.

    For Qwen pass prompt_think_open from prompt_has_open_think(actual_prompt_ids).
    EOS at the last allowed step reaches the cap but is not right-censored.
    A malformed think trace keeps its mechanical first-close counts and is
    explicitly flagged; callers must report anomalies, not silently drop them.
    """
    ids = list(ids)
    if not ids or any(not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in ids):
        raise Blocked('Expected nonempty original generated integer token IDs')
    if not isinstance(max_new_tokens, int) or not 0 < len(ids) <= max_new_tokens:
        raise Blocked('Invalid output length/cap')
    eos = set([] if eos_token_ids is None else [eos_token_ids] if isinstance(eos_token_ids,int) else eos_token_ids)
    ended = ids[-1] in eos
    capped = len(ids) == max_new_tokens
    anomalies = []
    if any(t in eos for t in ids[:-1]): anomalies.append('tokens_after_eos')
    if not ended and not capped: anomalies.append('unexpected_early_termination')
    result = {'output_tokens': len(ids), 'eos_terminated': ended,
              'output_cap_reached': capped, 'output_right_censored': capped and not ended,
              'finish_reason': 'eos' if ended else 'length' if capped else 'unexpected_early_termination',
              'length_partition_version': 'r3-original-token-v1',
              'lengths_are_observed': True}
    if model_key in ('llava7','mini8'):
        result.update(explanation_boundary(ids, tokenizer))
        result.update(reasoning_tokens=None, answer_tokens=None, structural_tokens=None,
                      thinking_boundary_found=None, reasoning_right_censored=None,
                      censoring='output_length' if capped and not ended else 'natural_eos' if ended else 'unexpected_early_termination')
    elif model_key in ('qwen8','qwen4'):
        if not isinstance(prompt_think_open,bool):
            raise Blocked('Need the actual prompt think-boundary state')
        opening,closing = _think_sequences(tokenizer)
        opens,closes = _matches(ids,opening),_matches(ids,closing)
        boundary = closes[0] if closes else None
        special = set(tokenizer.all_special_ids) | eos
        structural = {i for i,t in enumerate(ids) if t in special}
        for starts, seq in ((opens,opening),(closes,closing)):
            for start in starts: structural.update(range(start,start+len(seq)))
        if len(closes)>1: anomalies.append('multiple_think_close')
        if len(opens)>1: anomalies.append('multiple_think_open')
        if prompt_think_open and opens: anomalies.append('generated_think_open_while_prompt_open')
        if not prompt_think_open and not opens: anomalies.append('missing_think_open')
        if boundary is not None and opens and opens[0]>boundary: anomalies.append('think_open_after_close')
        if boundary is not None and not prompt_think_open and not any(i<boundary for i in opens):
            anomalies.append('close_without_prior_open')
        if ended and boundary is None: anomalies.append('eos_without_think_end')
        before_end = len(ids) if boundary is None else boundary
        after_start = len(ids) if boundary is None else boundary+len(closing)
        reasoning = sum(i not in structural for i in range(before_end))
        answer = sum(i not in structural for i in range(after_start,len(ids)))
        if reasoning+answer+len(structural)!=len(ids):
            raise Blocked('Non-exhaustive original token partition')
        result.update(reasoning_tokens=reasoning, answer_tokens=answer, structural_tokens=len(structural),
                      thinking_boundary_found=boundary is not None, thinking_boundary_offset=boundary,
                      think_close_offsets=closes, think_open_offsets=opens, prompt_think_open=prompt_think_open,
                      reasoning_right_censored=boundary is None and capped and not ended,
                      censoring='natural_think_end' if boundary is not None else
                      'length_before_think_end' if capped and not ended else
                      'eos_without_think_end' if ended else 'unexpected_early_termination',
                      final_text=_decode(tokenizer,ids[after_start:],visible=True),
                      explicit_explanation_tokens=None)
    else:
        raise Blocked('Unregistered model for token partitions')
    result.update(boundary_anomalies=anomalies,boundary_anomaly=bool(anomalies))
    return result
