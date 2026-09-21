"""VLM (image-text-to-text) export validation for Qwen4Exp (issue #99).

Ports `scripts/validate_ov_ir_vs_native_vlm.py` (already run manually and confirmed passing
during development - see FINDINGS.md) into the pytest suite: exports the vision-language IR,
then compares `OVModelForVisualCausalLM.generate()` against native `transformers`
`AutoModelForImageTextToText.generate()` on a synthetic image + text prompt, at both the
greedy-token and per-step-logit level.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen4_exp_fixture import tiny_qwen4_exp_dir  # noqa: F401


NUM_NEW_TOKENS = 12
MIN_COS_SIM = 0.99


@pytest.fixture(scope="session")
def vlm_processor(tiny_qwen4_exp_dir):
    return AutoProcessor.from_pretrained(tiny_qwen4_exp_dir, trust_remote_code=True)


@pytest.fixture(scope="session")
def vlm_inputs(vlm_processor):
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in one word."},
            ],
        }
    ]
    text_prompt = vlm_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return vlm_processor(text=[text_prompt], images=[image], return_tensors="pt")


@pytest.fixture(scope="session")
def ov_vlm_model_dir(tiny_qwen4_exp_dir, tmp_path_factory):
    from optimum.exporters.openvino import main_export

    output_dir = tmp_path_factory.mktemp("qwen4_exp") / "ov-tiny-qwen4-exp-vlm"
    main_export(
        model_name_or_path=str(tiny_qwen4_exp_dir),
        output=str(output_dir),
        task="image-text-to-text",
        trust_remote_code=True,
        stateful=True,
    )
    return output_dir


def test_ov_vlm_export_produces_ir(ov_vlm_model_dir):
    """The vision-language export produces a language-model IR (main sanity check that the
    export completed; the vision tower and embedding IRs are additional files alongside it)."""
    assert any(ov_vlm_model_dir.glob("openvino_language_model.xml")) or (
        ov_vlm_model_dir / "openvino_model.xml"
    ).exists()


def test_ov_ir_vlm_generate_matches_native(tiny_qwen4_exp_dir, ov_vlm_model_dir, vlm_inputs):
    """Tier-3: the exported VLM IR, exercised via the documented `.generate()` API, matches
    native greedy decoding exactly (with per-step logit cosine similarity as a secondary,
    stronger check) for a synthetic image + text prompt."""
    from optimum.intel import OVModelForVisualCausalLM

    native_model = AutoModelForImageTextToText.from_pretrained(tiny_qwen4_exp_dir, trust_remote_code=True, dtype=torch.float32)
    native_model.eval()
    with torch.no_grad():
        native_out = native_model.generate(
            **vlm_inputs,
            max_new_tokens=NUM_NEW_TOKENS,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    native_tokens = native_out.sequences[0, vlm_inputs["input_ids"].shape[1] :].tolist()

    ov_model = OVModelForVisualCausalLM.from_pretrained(ov_vlm_model_dir, trust_remote_code=True)
    ov_out = ov_model.generate(
        **vlm_inputs,
        max_new_tokens=NUM_NEW_TOKENS,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    ov_tokens = ov_out.sequences[0, vlm_inputs["input_ids"].shape[1] :].tolist()

    min_cos_overall = 1.0
    num_steps = min(len(native_out.scores), len(ov_out.scores))
    for step in range(num_steps):
        n = native_out.scores[step].numpy().astype(np.float64).reshape(-1)
        o = ov_out.scores[step].numpy().astype(np.float64).reshape(-1)
        finite = np.isfinite(n) & np.isfinite(o)
        cos_sim = (n[finite] @ o[finite]) / (np.linalg.norm(n[finite]) * np.linalg.norm(o[finite]) + 1e-12)
        min_cos_overall = min(min_cos_overall, cos_sim)

    assert min_cos_overall > MIN_COS_SIM, f"cos_sim degraded too much: {min_cos_overall}"
    assert native_tokens == ov_tokens, "Greedy token sequences diverged (VLM Tier-3)"
