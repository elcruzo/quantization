# Int8 + NF4 + NVFP4-style FP4

Weight quantization from primitives: integer 8-bit (tensor / channel / row / zero-point), **NF4** (QLoRA-style frozen-base API), and 4-bit E2M1 with hierarchical scales (NVFP4).

## Papers

- Dettmers et al., *LLM.int8()* / bitsandbytes — absmax Int8, vector-wise scales, outlier sensitivity.
- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* — affine (scale + zero-point).
- Dettmers et al., *QLoRA* — **NF4** (NormalFloat4) codebook + block absmax; QDoRA-style training freezes the same NF4 base under DoRA adapters (wired in PEFT folders, not here).
- NVIDIA, *NVFP4* (2025): E2M1 data, **block size 16**, FP8 E4M3 per-block scale, optional FP32 tensor scale.
- OCP MX formats: **MXFP4** uses block 32 and an E8M0 (power-of-two) scale.

## Int8

Symmetric: `scale = max|x| / 127`, `q = clip(round(x/s), -127, 127)`.
Per-channel: one scale per output row of a Linear weight (`Int8Weight`).
Per-row (LLM.int8 vector-wise): one scale per activation row.
Asymmetric: `scale = (max-min)/255`, `zp = round(-min/s)` (signed — a range like `[10, 20]` needs `zp < 0`), `x ≈ s·(q - zp)`.

## NF4 (QLoRA-style base API)

16-level codebook from normal quantiles (bitsandbytes `create_normal_map`), stored as indices — not a float encoding.
Blocks of **64** along the **last** axis; each block absmax-normalizes into `[-1, 1]`, then nearest NF4 level.
`NF4Weight.dequant()` returns the reconstructed weight for adapters to multiply against; codes stay frozen. This folder ships the codec — PEFT/LoRA packages own the training loop.

## FP4 / NVFP4

E2M1 magnitudes: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` plus a sign bit.

NVFP4-style reconstructs `x ≈ s_global * s_block * e2m1` with blocks along the **last** axis (never `ravel()`):

1. Per-block `s_raw = max|block| / 6`
2. `s_global = max(s_raw) / 448` so the block scales fit in E4M3
3. `s_block = quantize_e4m3(s_raw / s_global)` (E4M3 simulated as float)
4. codes = nearest E2M1 of `x / (s_global * s_block)`

On heavy-tailed / outlier tensors a single per-tensor int4 scale is set by the outliers and the mass around 0 collapses to zero. Block-16 hierarchical scales isolate the outliers.

E4M3 here is a **simulation** (3-bit mantissa, bias-7, clamp 448) — not a bit-exact NVIDIA converter.

## Papers on disk

- [`papers/dettmers-llm-int8-2022.pdf`](papers/dettmers-llm-int8-2022.pdf) — Dettmers et al. LLM.int8() (2022) ([arXiv:2208.07339](https://arxiv.org/abs/2208.07339))
- [`papers/dettmers-qlora-2023.pdf`](papers/dettmers-qlora-2023.pdf) — Dettmers et al. QLoRA / NF4 (2023) ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314))

## Run

```bash
python demo.py
python -m pytest test_quant.py -q
```
