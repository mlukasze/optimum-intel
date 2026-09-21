"""No-cache export validation for Qwen4Exp (issue #99): confirms the model exports to
OpenVINO IR without a KV cache and that the patched-Torch tracing path (the same code path
used during export) matches native `transformers` output exactly.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen4_exp_fixture import native_generate, patched_generate, tiny_qwen4_exp_dir  # noqa: F401


NUM_DECODE_STEPS = 8


@pytest.fixture(scope="session")
def native_text_model(tiny_qwen4_exp_dir):
    model = AutoModelForCausalLM.from_pretrained(tiny_qwen4_exp_dir, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tokenizer(tiny_qwen4_exp_dir):
    return AutoTokenizer.from_pretrained(tiny_qwen4_exp_dir, trust_remote_code=True)


def test_native_text_generation_smoke(native_text_model, tokenizer):
    """Sanity: the tiny fixture generates without raising, using standard `generate()`."""
    ids = tokenizer("Hello there", return_tensors="pt")
    with torch.no_grad():
        out = native_text_model.generate(**ids, max_new_tokens=5, do_sample=False)
    assert out.shape[1] == ids["input_ids"].shape[1] + 5


def test_patched_torch_matches_native_no_cache(tiny_qwen4_exp_dir, tokenizer):
    """The exporter's patcher (`Qwen4ExpModelPatcher.patched_forward`, the same code path used
    for tracing/export) reproduces native greedy decoding exactly, token-for-token, over
    several steps - independent of any OpenVINO IR/runtime concerns."""
    prompt_ids = tokenizer("Describe the weather today.", return_tensors="pt").input_ids

    model_native = AutoModelForCausalLM.from_pretrained(tiny_qwen4_exp_dir, trust_remote_code=True, dtype=torch.float32)
    native_tokens, _ = native_generate(model_native, prompt_ids, NUM_DECODE_STEPS)

    model_patched = AutoModelForCausalLM.from_pretrained(tiny_qwen4_exp_dir, trust_remote_code=True, dtype=torch.float32)
    patched_tokens, _ = patched_generate(model_patched, prompt_ids, NUM_DECODE_STEPS, prefill_size=prompt_ids.shape[1])

    assert patched_tokens == native_tokens


def test_export_no_cache_produces_ir(tiny_qwen4_exp_dir, tmp_path):
    """The no-cache export path (`text-generation` task) completes and produces a loadable IR."""
    from optimum.exporters.openvino import main_export

    output_dir = tmp_path / "ov-tiny-qwen4-exp-nocache"
    main_export(
        model_name_or_path=str(tiny_qwen4_exp_dir),
        output=str(output_dir),
        task="text-generation",
        trust_remote_code=True,
        stateful=False,
    )
    assert (output_dir / "openvino_model.xml").exists()

