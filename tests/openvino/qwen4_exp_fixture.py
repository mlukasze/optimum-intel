"""Shared tiny Qwen4Exp (Qwen/Qwen3.8-Flash-Next) fixture builder for the
`test_qwen4_exp_*.py` suite (issue #99, openvinotoolkit/omega).

Derives a tiny, architecturally-faithful config from the real `config.json` (metadata only,
no weight download) and overrides only size-related fields, mirroring the approach used to
produce the reference fixture during development (see the linked issue for the full report).
Kept deliberately small (4 hidden layers - one period of the real model's repeating
[linear, linear, linear, qwen_sparse_attention] pattern, one MoE + one PLE layer) so the whole
suite runs quickly, while still exercising every architectural branch: linear attention, QSA
sparse attention, MoE routing, PLE n-gram embedding, and the vision tower.
"""

import copy
import json

import pytest
import torch


MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
SEED = 1234

# Matches `create_tiny_model.py` (the fixture already validated end-to-end during development -
# see FINDINGS.md) exactly, rather than an independently hand-trimmed config: an earlier,
# smaller variant of this fixture (4 layers, hidden_size=64) hit a degenerate zero-sized
# `Assign` state descriptor specific to that config's dimensions, unrelated to the actual
# architecture support being tested. Reusing the proven configuration avoids chasing fixture-
# specific artifacts outside the ticket's scope.
TEXT_CONFIG_OVERRIDES = {
    "hidden_size": 128,
    "num_hidden_layers": 16,
    "full_attention_interval": 4,
    "head_dim": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "indexer_budget": 32,
    "indexer_compress_ratio": 4,
    "indexer_n_heads": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 32,
    "ple_layer_ids": [2],
    "ple_embed_dim": 128,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ngram_vocab_size_base": 2000,
    "split_ngram_parts": 8,
    "make_ngram_vocab_size_divisible_by": 8,
    "seed": SEED,
    "hc_count": 4,
    "hc_lowrank": 16,
    "mtp_num_hidden_layers": 1,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
    # vocab_size deliberately NOT overridden - kept at the real value (chat template / special
    # tokens rely on ids that only make sense against the real tokenizer).
}

# Repeats the real model's [linear, linear, linear, qwen_sparse_attention] pattern x4 = 16 layers.
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 4

VISION_CONFIG_OVERRIDES = {
    "depth": 2,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_heads": 4,
    "out_hidden_size": 128,  # must equal tiny text hidden_size
    "num_position_embeddings": 128,
    "deepstack_visual_indexes": [],
}


def _build_tiny_config(real_config):
    from transformers import Qwen4ExpConfig  # noqa: F401  (imported for clarity / type context)

    tiny_config = copy.deepcopy(real_config)

    tiny_config.text_config.layer_types = list(LAYER_TYPES)
    for key, value in TEXT_CONFIG_OVERRIDES.items():
        setattr(tiny_config.text_config, key, value)
    tiny_config.text_config.__post_init__()

    for key, value in VISION_CONFIG_OVERRIDES.items():
        setattr(tiny_config.vision_config, key, value)

    tiny_config.text_config.dtype = "float32"
    tiny_config.text_config.mamba_ssm_dtype = "float32"
    tiny_config.vision_config.dtype = "float32"
    if hasattr(tiny_config, "dtype"):
        tiny_config.dtype = "float32"

    # mrope_section must sum to head_dim * partial_rotary_factor // 2; scale the real
    # proportions down to the tiny rotary width and fix up rounding drift.
    rotary_dim_half = int(
        tiny_config.text_config.head_dim * tiny_config.text_config.rope_parameters["partial_rotary_factor"] // 2
    )
    real_section = real_config.text_config.rope_parameters["mrope_section"]
    real_sum = sum(real_section)
    scaled = [max(1, round(part * rotary_dim_half / real_sum)) for part in real_section]
    scaled[0] += rotary_dim_half - sum(scaled)
    tiny_config.text_config.rope_parameters["mrope_section"] = scaled

    return tiny_config


def build_tiny_qwen4_exp(out_dir, seed: int = SEED):
    """Build (or reuse a cached) tiny Qwen4Exp checkpoint at `out_dir` and return its path."""
    from transformers import AutoProcessor, AutoTokenizer, Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_marker = out_dir / ".tiny_model_cache_marker.json"
    if cache_marker.exists():
        marker = json.loads(cache_marker.read_text())
        if marker.get("model_id") == MODEL_ID and marker.get("seed") == seed:
            return out_dir

    real_config = Qwen4ExpConfig.from_pretrained(MODEL_ID)
    tiny_config = _build_tiny_config(real_config)

    torch.manual_seed(seed)
    model = Qwen4ExpForConditionalGeneration(tiny_config).to(torch.float32)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 200_000_000, f"Tiny test fixture unexpectedly large: {n_params / 1e6:.1f}M params"
    model.save_pretrained(out_dir)

    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(out_dir)
    AutoProcessor.from_pretrained(MODEL_ID).save_pretrained(out_dir)

    cache_marker.write_text(json.dumps({"model_id": MODEL_ID, "seed": seed, "n_params": n_params}))
    return out_dir


@pytest.fixture(scope="session")
def tiny_qwen4_exp_dir(tmp_path_factory):
    """Session-scoped fixture: build the tiny Qwen4Exp checkpoint once, reuse across all
    `test_qwen4_exp_*.py` modules."""
    out_dir = tmp_path_factory.mktemp("qwen4_exp") / "tiny-qwen4-exp"
    return build_tiny_qwen4_exp(out_dir)


def get_export_config(model, task="text-generation-with-past"):
    from optimum.exporters.tasks import TasksManager

    ctor = TasksManager.get_exporter_config_constructor(
        model=model, exporter="openvino", task=task, library_name="transformers"
    )
    return ctor(model.config, task=task)


def make_empty_cache_params(export_config, batch_size):
    """Build a real (not None) initial `cache_params` list representing "no history yet", using
    the same `Qwen4ExpDummyPastKeyValuesGenerator` the exporter itself uses with
    `sequence_length=0` so the growing key/value/indexer slots start empty."""
    from optimum.exporters.openvino.input_generators import Qwen4ExpDummyPastKeyValuesGenerator

    generator = Qwen4ExpDummyPastKeyValuesGenerator(
        task="text-generation-with-past",
        normalized_config=export_config._normalized_config,
        batch_size=batch_size,
        sequence_length=0,
    )
    cache_params = generator.generate("cache_params")
    for tensor in cache_params:
        tensor.zero_()
    return cache_params


def native_generate(model, input_ids, n_steps):
    """Greedy decode via the standard HF `Cache` object, one token at a time."""
    model.eval()
    tokens, logits_list = [], []
    cur_input_ids = input_ids
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand(input_ids.shape[0], -1)
    past_key_values = None
    with torch.no_grad():
        for _ in range(n_steps):
            out = model(
                input_ids=cur_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            tokens.append(next_token.squeeze(-1).tolist())  # one entry per batch element
            logits_list.append(logits.clone())
            past_key_values = out.past_key_values
            cur_input_ids = next_token
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            position_ids = position_ids[:, -1:] + 1
    return tokens, logits_list


def patched_generate(model, input_ids, n_steps, prefill_size=1):
    """Greedy decode via the exporter's `Qwen4ExpModelPatcher.patched_forward`, using the flat
    `cache_params` list contract used for OpenVINO export/tracing. `prefill_size` controls how
    many of the prompt tokens are fed in the first call (>1 simulates a multi-token prefill
    step, matching real inference; 1 simulates the export dummy shape)."""
    from optimum.exporters.openvino.model_patcher import Qwen4ExpModelPatcher

    export_config = get_export_config(model)
    patcher = Qwen4ExpModelPatcher(export_config, model, {})
    tokens, logits_list = [], []
    with patcher:
        cache_params = make_empty_cache_params(export_config, batch_size=input_ids.shape[0])
        prefill_size = min(prefill_size, input_ids.shape[1])
        cur_input_ids = input_ids[:, :prefill_size]
        remaining = input_ids[:, prefill_size:]
        attention_mask = torch.ones_like(input_ids[:, :prefill_size])
        position_ids = torch.arange(prefill_size).unsqueeze(0).expand(input_ids.shape[0], -1)
        step = 0
        with torch.no_grad():
            while True:
                out = patcher.patched_forward(
                    input_ids=cur_input_ids,
                    attention_mask=attention_mask,
                    cache_params=cache_params,
                    position_ids=position_ids,
                )
                cache_params = out["present_key_values"]
                if remaining.shape[1] > 0:
                    # still feeding the (multi-token) prompt; don't score these positions
                    cur_input_ids = remaining[:, :1]
                    position_ids = position_ids[:, -1:] + 1
                    attention_mask = torch.cat([attention_mask, torch.ones_like(cur_input_ids)], dim=1)
                    remaining = remaining[:, 1:]
                    continue
                logits = out["logits"][:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)
                tokens.append(next_token.squeeze(-1).tolist())  # one entry per batch element
                logits_list.append(logits.clone())
                step += 1
                if step >= n_steps:
                    break
                cur_input_ids = next_token
                position_ids = position_ids[:, -1:] + 1
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
    return tokens, logits_list
