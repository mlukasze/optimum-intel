# SeamlessM4T-v2 — OpenVINO Conversion

This directory contains a script to convert `facebook/seamless-m4t-v2-large` to
OpenVINO IR format. The model is not yet natively supported by optimum-intel exporters,
so direct `ov.convert_model()` is used with a minimal wrapper.

## Requirements

```bash
pip install openvino transformers torch
```

## Conversion

```bash
python convert_to_ov.py \
  --model-id facebook/seamless-m4t-v2-large \
  --output-dir ./ov_models/seamless-m4t-v2 \
  --task t2tt
```

## Verified Results

| Task | Params | Precision | CPU | GPU | Accuracy (vs PyTorch) |
|------|--------|-----------|-----|-----|-----------------------|
| T2TT | 1.37B  | FP16      | ✅  | ✅  | 0.01% deviation (cosine) |

## Notes

- **Workaround**: `EncoderDecoderCache` is not JIT-traceable. The `T2TTWrapper` class
  passes `use_cache=False, return_dict=False` to get only logits as output.
- License: cc-by-nc-4.0 (non-commercial use only)
- Native optimum-intel support would require adding `seamless_m4t_v2` to `TasksManager`.
