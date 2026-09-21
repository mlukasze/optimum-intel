"""Component-level parity tests for the Qwen4Exp OpenVINO export patches (issue #99).

Validates the trace-friendly, fully vectorized `qwen4_exp_traceable_qsa_indexer_forward`
(no Python-level shape branching, safe to export) bit-for-bit against the stock nested-loop
`Qwen4ExpTextQSAIndexer.forward` reference shipped in `transformers`, across a matrix of
batch sizes / query lengths / kv lengths chosen to exercise every boundary condition relevant
to cached decoding (kv_length shorter than / equal to / not a multiple of `compress_ratio`,
single-token decode steps, etc).

Restriction (documented, matches production usage): this vectorized form assumes standard
causal decoding with no padding holes in the attention mask (query i can see exactly kv
positions [0 .. cache_offset + i]), which covers the actual generation task path (chat /
image+text prompts, greedy decode) this pipeline targets.
"""

import math

import pytest
import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextConfig,
    Qwen4ExpTextQSAIndexer,
    Qwen4ExpTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

from optimum.exporters.openvino.model_patcher import qwen4_exp_traceable_qsa_indexer_forward


def _make_causal_mask(batch_size, seq_length, kv_length, dtype, as_bool):
    past = kv_length - seq_length
    q_abs = past + torch.arange(seq_length)
    kv_ids = torch.arange(kv_length)
    visible = kv_ids.view(1, kv_length) <= q_abs.view(seq_length, 1)
    visible = visible.view(1, 1, seq_length, kv_length).expand(batch_size, 1, seq_length, kv_length)
    if as_bool:
        return visible.clone()
    mask = torch.zeros(batch_size, 1, seq_length, kv_length, dtype=dtype)
    return mask.masked_fill(~visible, torch.finfo(dtype).min)


def _stock_qsa_indexer_forward(indexer, hidden_states, position_embeddings, attention_mask, raw_keys_full):
    """Direct transcription of `Qwen4ExpTextQSAIndexer.forward`, taking the already
    cache-updated `raw_keys_full` directly so it can be exercised standalone."""
    batch_size, seq_length, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_length, -1, indexer.index_head_dim)
    full_cos, full_sin = position_embeddings
    current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

    qk = indexer.index_qk_proj(hidden_states)
    q, _ = torch.split(
        qk, [indexer.index_n_heads * indexer.index_head_dim, indexer.index_kv_heads * indexer.index_head_dim], dim=-1
    )
    q = indexer.q_layernorm(q.reshape(*hidden_shape))
    q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

    visible_token_indices = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    selected_token_indices = torch.full(
        (batch_size, seq_length, indexer.token_budget + indexer.compress_ratio - 1),
        -1,
        dtype=torch.int32,
    )
    for batch_idx in range(batch_size):
        for query_idx in range(seq_length):
            local_visible_indices = torch.nonzero(
                visible_token_indices[batch_idx, 0, query_idx], as_tuple=False
            ).flatten()
            num_complete_blocks = local_visible_indices.shape[-1] // indexer.compress_ratio
            if num_complete_blocks > 0:
                block_token_indices = local_visible_indices[: num_complete_blocks * indexer.compress_ratio].view(
                    num_complete_blocks, indexer.compress_ratio
                )
                key_groups = raw_keys_full[batch_idx].index_select(0, block_token_indices.flatten())
                key_groups = key_groups.view(*block_token_indices.shape, indexer.index_head_dim)
                pooled_keys = indexer.k_layernorm(key_groups.float().mean(dim=1).to(raw_keys_full.dtype))
                group_starts = block_token_indices[:, 0]
                block_key_states = apply_rotary_pos_emb(
                    pooled_keys.unsqueeze(1),
                    cos=full_cos[batch_idx].index_select(0, group_starts),
                    sin=full_sin[batch_idx].index_select(0, group_starts),
                ).squeeze(1)
                scores = torch.matmul(
                    q[batch_idx, query_idx].float(), block_key_states.float().transpose(-1, -2)
                ).transpose(-1, -2)
                scores = torch.relu(scores).sum(dim=-1) / math.sqrt(indexer.index_head_dim)
                selected_block_indices = scores.topk(min(indexer.block_topk, num_complete_blocks), dim=0).indices
                selected_tokens = block_token_indices.index_select(0, selected_block_indices).flatten()
            else:
                selected_tokens = torch.tensor([])
            tail = local_visible_indices[num_complete_blocks * indexer.compress_ratio :]
            selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
            selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

    kv_length = attention_mask.shape[-1]
    selected_token_mask = torch.zeros(
        (*selected_token_indices.shape[:-1], kv_length + 1), dtype=torch.bool
    )
    scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
    selected_token_mask = selected_token_mask.scatter(-1, scatter_indices.long(), True)[..., :kv_length].unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        selected_token_mask = torch.where(selected_token_mask, attention_mask.new_zeros(()), min_dtype)
    return selected_token_mask


class _FakeIndexerCache:
    """Minimal stand-in for the real export-time cache object's `update_indexer` contract
    (`optimum/exporters/openvino/model_patcher.py`): accumulates per-layer indexer key states
    along the sequence dimension and returns the full accumulated tensor, so the *real* traceable
    indexer function can be exercised with an arbitrary amount of pre-existing cached history."""

    def __init__(self):
        self._cache = {}

    def update_indexer(self, indexer_key_states, layer_idx):
        if layer_idx not in self._cache:
            self._cache[layer_idx] = indexer_key_states
        else:
            self._cache[layer_idx] = torch.cat([self._cache[layer_idx], indexer_key_states], dim=1)
        return self._cache[layer_idx]


def _traceable_qsa_indexer_forward_with_history(indexer, hidden_states_full, position_embeddings, attention_mask, past_length):
    """Call the *real* `qwen4_exp_traceable_qsa_indexer_forward` the same way the exported graph
    does across two incremental-decoding steps: `past_length` tokens are fed first (their own
    self-contained causal step, populating a fake cache), then the remaining (current) tokens are
    fed as `hidden_states` with the *full* accumulated position embeddings - so keys for the
    historical part come from the cache (`update_indexer`) instead of being independently
    sampled, matching production semantics exactly (keys are always derived from `hidden_states`
    via `index_qk_proj`, never passed in directly)."""
    full_cos, full_sin = position_embeddings
    if past_length == 0:
        return qwen4_exp_traceable_qsa_indexer_forward(indexer, hidden_states_full, position_embeddings, attention_mask, None)

    cache = _FakeIndexerCache()
    batch_size = hidden_states_full.shape[0]
    past_cos, past_sin = full_cos[:, :past_length, :], full_sin[:, :past_length, :]
    past_mask = _make_causal_mask(batch_size, past_length, past_length, torch.float32, as_bool=False)
    qwen4_exp_traceable_qsa_indexer_forward(indexer, hidden_states_full[:, :past_length], (past_cos, past_sin), past_mask, cache)

    return qwen4_exp_traceable_qsa_indexer_forward(
        indexer, hidden_states_full[:, past_length:], position_embeddings, attention_mask, cache
    )


def _indexer_config():
    return Qwen4ExpTextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=32,
        indexer_compress_ratio=4,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        layer_types=["qwen_sparse_attention"],
        num_hidden_layers=1,
        max_position_embeddings=512,
    )


# batch_size, kv_length, seq_length (query length); covers kv_length shorter than / equal to /
# not a multiple of compress_ratio=4, and both single-token-decode and multi-token-prefill.
BOUNDARY_SCENARIOS = [
    (batch_size, kv_length, seq_length)
    for batch_size in (1, 2)
    for kv_length in (1, 3, 4, 5, 7, 8, 9, 15, 16, 17)
    for seq_length in (1, min(kv_length, 3))
    if seq_length <= kv_length
]


@pytest.mark.parametrize("as_bool", [True, False])
@pytest.mark.parametrize("batch_size,kv_length,seq_length", BOUNDARY_SCENARIOS)
def test_qsa_indexer_vectorized_matches_stock_sequential(batch_size, kv_length, seq_length, as_bool):
    """Sequential (plain arange) position_ids - the standard causal-decode case. Exercises the
    *real* `qwen4_exp_traceable_qsa_indexer_forward` used by the exporter (via a fake cache
    object providing the `update_indexer` contract) against the stock nested-loop reference,
    fed the exact same full-sequence keys so both see identical projected key values."""
    torch.manual_seed(0)
    config = _indexer_config()
    indexer = Qwen4ExpTextQSAIndexer(config, layer_idx=0).eval()
    rotary = Qwen4ExpTextRotaryEmbedding(config)
    past_length = kv_length - seq_length

    hidden_states_full = torch.randn(batch_size, kv_length, config.hidden_size)
    attention_mask = _make_causal_mask(batch_size, seq_length, kv_length, torch.float32, as_bool)

    pos = torch.arange(kv_length).unsqueeze(0).expand(batch_size, -1)
    position_ids = pos.unsqueeze(0).expand(3, -1, -1).clone()
    cos, sin = rotary(hidden_states_full, position_ids)

    # Reference keys: derive from the same projection the traceable function uses internally,
    # over the full sequence, so both implementations start from identical key values.
    qk_full = indexer.index_qk_proj(hidden_states_full)
    _, token_k_full = torch.split(
        qk_full, [indexer.index_n_heads * indexer.index_head_dim, indexer.index_kv_heads * indexer.index_head_dim], dim=-1
    )
    raw_keys_full = token_k_full.reshape(batch_size, kv_length, indexer.index_kv_heads, indexer.index_head_dim).squeeze(2)

    ref = _stock_qsa_indexer_forward(indexer, hidden_states_full[:, past_length:], (cos, sin), attention_mask, raw_keys_full)
    vec = _traceable_qsa_indexer_forward_with_history(indexer, hidden_states_full, (cos, sin), attention_mask, past_length)

    assert ref.shape == vec.shape
    torch.testing.assert_close(ref, vec, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("batch_size,kv_length,seq_length", BOUNDARY_SCENARIOS)
def test_qsa_indexer_vectorized_matches_stock_non_sequential(batch_size, kv_length, seq_length):
    """Non-sequential ('VLM-like') position_ids: each M-RoPE row is an independent random
    permutation with repeats, mimicking a 2D vision grid's row/col position channels. This is
    the regression test for a previously-fixed bug where the vectorized indexer recomputed
    each block's RoPE angle assuming `group_start == position_id` (exact for plain sequential
    input, silently wrong otherwise) instead of gathering the real per-position value."""
    torch.manual_seed(0)
    config = _indexer_config()
    indexer = Qwen4ExpTextQSAIndexer(config, layer_idx=0).eval()
    rotary = Qwen4ExpTextRotaryEmbedding(config)
    past_length = kv_length - seq_length

    hidden_states_full = torch.randn(batch_size, kv_length, config.hidden_size)
    attention_mask = _make_causal_mask(batch_size, seq_length, kv_length, torch.float32, as_bool=False)

    position_ids = torch.stack(
        [torch.stack([torch.randperm(kv_length * 3)[:kv_length] for _ in range(batch_size)]) for _ in range(3)]
    )
    cos, sin = rotary(hidden_states_full, position_ids)

    qk_full = indexer.index_qk_proj(hidden_states_full)
    _, token_k_full = torch.split(
        qk_full, [indexer.index_n_heads * indexer.index_head_dim, indexer.index_kv_heads * indexer.index_head_dim], dim=-1
    )
    raw_keys_full = token_k_full.reshape(batch_size, kv_length, indexer.index_kv_heads, indexer.index_head_dim).squeeze(2)

    ref = _stock_qsa_indexer_forward(indexer, hidden_states_full[:, past_length:], (cos, sin), attention_mask, raw_keys_full)
    vec = _traceable_qsa_indexer_forward_with_history(indexer, hidden_states_full, (cos, sin), attention_mask, past_length)

    assert ref.shape == vec.shape
    torch.testing.assert_close(ref, vec, atol=1e-6, rtol=1e-5)
