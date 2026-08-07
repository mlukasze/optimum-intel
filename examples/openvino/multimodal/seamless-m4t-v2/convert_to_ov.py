"""
OpenVINO Conversion Script for facebook/seamless-m4t-v2-large

This script converts SeamlessM4T-v2 to OpenVINO IR format using direct
ov.convert_model() since the model is not yet supported by optimum-intel exporters.

Usage:
    python convert_to_ov.py --model-id facebook/seamless-m4t-v2-large --output-dir ./ov_models/

Supported tasks:
    - t2tt: Text-to-Text Translation
"""
import argparse
import os
from pathlib import Path
import torch
import openvino as ov


class T2TTWrapper(torch.nn.Module):
    """Wraps SeamlessM4Tv2ForTextToText for OV conversion.
    
    Workaround for EncoderDecoderCache not being serializable by PyTorch JIT tracer.
    Uses use_cache=False and return_dict=False to return only logits.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=False,
        )
        return out[0]  # logits only


def convert_t2tt(model_id: str, output_dir: Path, dtype: torch.dtype = torch.float16):
    """Convert SeamlessM4T-v2 text-to-text model to OpenVINO IR."""
    from transformers import SeamlessM4Tv2ForTextToText

    print(f"Loading {model_id} (T2TT task) in {'FP16' if dtype == torch.float16 else 'FP32'}...")
    model = SeamlessM4Tv2ForTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded: {n_params:.2f}B parameters")

    wrapped = T2TTWrapper(model)
    wrapped.eval()

    # Dummy inputs for tracing
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(4, min(vocab_size, 256000), (1, 32), dtype=torch.long)
    attention_mask = torch.ones(1, 32, dtype=torch.long)
    decoder_input_ids = torch.randint(4, min(vocab_size, 256000), (1, 5), dtype=torch.long)

    print("Converting to OpenVINO IR...")
    ov_model = ov.convert_model(
        wrapped,
        example_input=(input_ids, attention_mask, decoder_input_ids),
    )

    out_path = output_dir / "t2tt-fp16" / "model.xml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(out_path))
    print(f"Saved: {out_path}")
    return str(out_path.parent)


def verify_inference(model_xml: str, device: str = "CPU"):
    """Verify the exported model runs inference correctly."""
    import numpy as np

    core = ov.Core()
    print(f"\nVerifying inference on {device}...")
    model = core.read_model(model_xml)
    compiled = core.compile_model(model, device)
    request = compiled.create_infer_request()

    input_ids = np.random.randint(4, 256000, (1, 32), dtype=np.int64)
    attention_mask = np.ones((1, 32), dtype=np.int64)
    decoder_input_ids = np.random.randint(4, 256000, (1, 5), dtype=np.int64)

    result = request.infer({
        model.inputs[0]: input_ids,
        model.inputs[1]: attention_mask,
        model.inputs[2]: decoder_input_ids,
    })
    output = list(result.values())[0]
    print(f"[{device}] Inference OK — output shape: {output.shape}, dtype: {output.dtype}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert SeamlessM4T-v2 to OpenVINO IR")
    parser.add_argument("--model-id", default="facebook/seamless-m4t-v2-large")
    parser.add_argument("--output-dir", default="./ov_models/seamless-m4t-v2")
    parser.add_argument("--task", default="t2tt", choices=["t2tt"])
    parser.add_argument("--device", default="CPU", help="Verification device (CPU, GPU)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "t2tt":
        model_dir = convert_t2tt(args.model_id, output_dir)
        verify_inference(str(Path(model_dir) / "model.xml"), args.device)

    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()
