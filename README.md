# Int8 + NVFP4-style FP4

Weight quantization: integer 8-bit (tensor / channel / zero-point) and 4-bit E2M1 with hierarchical scales.

## Papers

- Dettmers et al., *LLM.int8()* / bitsandbytes — absmax Int8, outlier sensitivity.
- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* — affine (scale + zero-point).
- NVIDIA, *NVFP4* (2025): E2M1 data, **block size 16**, FP8 E4M3 per-block scale, optional FP32 tensor scale.
- OCP MX formats: **MXFP4** uses block 32 and an E8M0 (power-of-two) scale.

## Int8

Symmetric: `scale = max|x| / 127`, `q = clip(round(x/s), -127, 127)`.
Per-channel: one scale per output row of a Linear weight.
Asymmetric: `scale = (max-min)/255`, `zp = round(-min/s)` (signed — a range like `[10, 20]` needs `zp < 0`), `x ≈ s·(q - zp)`.
A shifted tensor reconstructs far better with that affine map than with symmetric Int8, which spends codes on unused negatives.

## FP4

E2M1 magnitudes: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` plus a sign bit.

NVFP4-style reconstructs `x ≈ s_global * s_block * e2m1`:

1. Per-block `s_raw = max|block| / 6`
2. `s_global = max(s_raw) / 448` so the block scales fit in E4M3
3. `s_block = quantize_e4m3(s_raw / s_global)` (E4M3 simulated as float)
4. codes = nearest E2M1 of `x / (s_global * s_block)`

On heavy-tailed / outlier tensors a single per-tensor int4 scale is set by the outliers and the mass around 0 collapses to zero. Block-16 hierarchical scales isolate the outliers.

E4M3 here is a **simulation** (3-bit mantissa, bias-7, clamp 448) — not a bit-exact NVIDIA converter.

## Run

```bash
python demo.py
python -m pytest test_quant.py -q
```
