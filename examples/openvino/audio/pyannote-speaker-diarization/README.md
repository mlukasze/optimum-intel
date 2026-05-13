# pyannote Speaker Diarization — OpenVINO Conversion

Converts the component models of `pyannote/speaker-diarization-community-1` to OpenVINO IR.

## Requirements

```bash
pip install openvino pyannote.audio torch
huggingface-cli login  # required for gated model access
```

## Usage

```bash
python convert_to_ov.py --output-dir ./ov_models/pyannote-diarization
```

## Models Exported

| Model | Task | Input | Accuracy (CPU) |
|-------|------|-------|----------------|
| `pyannote/segmentation-3.0` | Speaker segmentation | `(B, 1, T)` raw waveform | 0.000% deviation |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | Speaker embeddings (backbone) | `(B, 80, T)` mel-spectrogram | 0.004% deviation |

## Notes

- **Segmentation**: Exported via `torch.onnx.export` (legacy trace mode) + `ov.convert_model`.
  `SegWrapper` returns the `"speaker"` key from the dict output.
- **Embedding backbone**: `torch.vmap` in `compute_fbank` prevents full model export.
  The `resnet` child module accepts pre-computed mel-spectrogram features directly.
- License: CC-BY-4.0 (attribution required)
