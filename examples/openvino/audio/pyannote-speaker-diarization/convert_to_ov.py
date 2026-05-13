"""
OpenVINO Conversion Script for pyannote/speaker-diarization-community-1

Converts the two component models to OpenVINO IR:
1. Segmentation model (pyannote/segmentation-3.0): speaker segmentation
2. Embedding model backbone (pyannote/wespeaker-voxceleb-resnet34-LM): speaker embeddings

Requirements:
    pip install openvino pyannote.audio torch

Note:
    The full pipeline model (pyannote/speaker-diarization-community-1) is gated on
    HuggingFace and requires authentication. Its components are exported separately.

Usage:
    python convert_to_ov.py --output-dir ./ov_models/
"""
import argparse
import io
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import openvino as ov


class SegWrapper(nn.Module):
    """Wraps pyannote segmentation model to return a single tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        out = self.model(waveform)
        return out["speaker"] if isinstance(out, dict) else out


def convert_segmentation(output_dir: Path) -> str:
    """Convert pyannote/segmentation-3.0 to OpenVINO IR via ONNX.

    Args:
        output_dir: Directory to save the OV IR model.

    Returns:
        Path to saved model.xml.
    """
    from pyannote.audio import Model as PAModel

    print("Loading pyannote/segmentation-3.0...")
    model = SegWrapper(PAModel.from_pretrained("pyannote/segmentation-3.0"))
    model.eval()

    # 10-second chunk at 16 kHz
    dummy = torch.randn(1, 1, 160000)
    buf = io.BytesIO()
    torch.onnx.export(
        model, (dummy,), buf, opset_version=14,
        input_names=["waveform"], output_names=["segmentation"],
        dynamic_axes={"waveform": {0: "batch"}},
        dynamo=False,
    )
    buf.seek(0)

    ov_model = ov.convert_model(buf)
    out_path = output_dir / "segmentation-fp32" / "model.xml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(out_path))
    print(f"Segmentation model saved: {out_path.parent}")
    return str(out_path)


def convert_embedding_backbone(output_dir: Path) -> str:
    """Convert the ResNet34 backbone of pyannote/wespeaker-voxceleb-resnet34-LM.

    Note: The full model uses torch.vmap for mel-filterbank computation which is
    not exportable. The ResNet backbone accepts pre-computed mel-spectrogram
    features (80 mel-bins × N time-frames) and outputs intermediate features.
    Use forward_frames() for feature-only inference externally.

    Args:
        output_dir: Directory to save the OV IR model.

    Returns:
        Path to saved model.xml.
    """
    from pyannote.audio import Model as PAModel

    print("Loading pyannote/wespeaker-voxceleb-resnet34-LM...")
    full_model = PAModel.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    backbone = full_model.resnet
    backbone.eval()

    # Mel-spectrogram input: 80 bins × 200 frames
    dummy = torch.randn(1, 80, 200)
    ov_model = ov.convert_model(backbone, example_input=(dummy,))
    out_path = output_dir / "embedding-resnet-fp32" / "model.xml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(out_path))
    print(f"Embedding backbone saved: {out_path.parent}")
    return str(out_path)


def verify(model_xml: str, dummy_input: np.ndarray, device: str = "CPU") -> bool:
    core = ov.Core()
    compiled = core.compile_model(model_xml, device)
    output = compiled(dummy_input)[0]
    print(f"  [{device}] shape={output.shape} dtype={output.dtype} ✅")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./ov_models/pyannote-diarization")
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_xml = convert_segmentation(output_dir)
    print("Verifying segmentation...")
    verify(seg_xml, np.random.randn(1, 1, 160000).astype(np.float32), args.device)

    emb_xml = convert_embedding_backbone(output_dir)
    print("Verifying embedding backbone...")
    verify(emb_xml, np.random.randn(1, 80, 200).astype(np.float32), args.device)

    print("\n✅ All models exported and verified!")


if __name__ == "__main__":
    main()
