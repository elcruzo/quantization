# Int8 + NF4 + NVFP4-style FP4

Weight quantization from primitives: integer 8-bit (tensor / channel / row / zero-point), **NF4** (QLoRA-style frozen-base API), and 4-bit E2M1 with hierarchical scales (NVFP4).

## Papers

- Dettmers et al., *LLM.int8()* / bitsandbytes — absmax Int8, vector-wise scales, outlier sensitivity.
- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* — affine (scale + zero-point).
- Dettmers et al., *QLoRA* — **NF4** (NormalFloat4) codebook + block absmax; QDoRA-style training freezes the same NF4 base under DoRA adapters. This folder ships the NF4 codec; QDoRA training lives in the PEFT folders.
- NVIDIA, *NVFP4* (2025): E2M1 data, **block size 16**, FP8 E4M3 per-block scale, optional FP32 tensor scale.
- OCP MX formats: **MXFP4** uses block 32 and an E8M0 (power-of-two) scale.

## Int8

Symmetric: `scale = max|x| / 127`, `q = clip(round(x/s), -127, 127)`.
Per-channel: one scale per output row of a Linear weight (`Int8Weight`).
Per-row (LLM.int8 vector-wise): one scale per activation row.
Asymmetric: `scale = (max-min)/255`, `zp = round(-min/s)` (signed — a range like `[10, 20]` needs `zp < 0`), `x ≈ s·(q - zp)`.

## NF4 (QLoRA-style base API)

16-level codebook from normal quantiles (bitsandbytes `create_normal_map`), stored as indices.
Blocks of **64** along the **last** axis; each block absmax-normalizes into `[-1, 1]`, then nearest NF4 level.
`NF4Weight.dequant()` returns the reconstructed weight for adapters to multiply against; codes stay frozen. This folder ships the codec — PEFT/LoRA packages own the training loop.

## FP4 / NVFP4

E2M1 magnitudes: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` plus a sign bit.

NVFP4-style reconstructs $x \approx s_{\mathrm{global}} \cdot s_{\mathrm{block}} \cdot e2m1$ with blocks along the **last** axis:

1. Per-block `s_raw = max|block| / 6`
2. `s_global = max(s_raw) / 448` so the block scales fit in E4M3
3. `s_block = quantize_e4m3(s_raw / s_global)` (E4M3 grid in float32)
4. codes = nearest E2M1 of `x / (s_global * s_block)`

On heavy-tailed / outlier tensors a single per-tensor int4 scale is set by the outliers and the mass around 0 collapses to zero. Block-16 hierarchical scales isolate the outliers.

E4M3 scales live in float32 on the E4M3 numeric grid (3-bit mantissa, bias-7, clamp 448).

## Papers on disk

- [`papers/dettmers-llm-int8-2022.pdf`](papers/dettmers-llm-int8-2022.pdf) — Dettmers et al. LLM.int8() (2022) ([arXiv:2208.07339](https://arxiv.org/abs/2208.07339))
- [`papers/dettmers-qlora-2023.pdf`](papers/dettmers-qlora-2023.pdf) — Dettmers et al. QLoRA / NF4 (2023) ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314))

## Compared to bitsandbytes / NVFP4

**What you learn here:**
- Int8 tensor/channel/asymmetric codecs from absmax / affine formulas
- NF4 block codebook (QLoRA-style) and NVFP4 E2M1 + E4M3 hierarchical scales
- Why block scales beat per-tensor int4 on outlier-heavy tensors

| | This repo | bitsandbytes / NVIDIA NVFP4 |
|---|---|---|
| Kernels | NumPy reference | CUDA GEMM / Blackwell |
| NF4 | Codec + `dequant()` API | QLoRA frozen-base training |
| FP4 | E4M3 grid in float32 | Hardware converters |

### Numbers (2026-08-16, Darwin 25.5.0 arm64 / Apple M5)

| Metric | This repo | Baseline | Source |
|---|---|---|---|
| Weight MSE NF4 | $7.1{\times}10^{-6}$ | — | `python main.py` |
| Outlier MSE NVFP4 | $5.7{\times}10^{-5}$ | ≫ int4 $6.4{\times}10^{-3}$ here | same |
| Memory vs FP16 | 4-bit storage (~4×) | ~4× NF4 vs FP16 base | Dettmers QLoRA 2023 |

```bash
python main.py
```

## Run

```bash
python main.py
python -m pytest test_quant.py -q
```
