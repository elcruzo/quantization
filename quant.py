"""Int8 (LLM.int8-style), NF4, and NVFP4 hierarchical E2M1 FP4.

Int8: per-tensor and per-channel/row symmetric absmax, plus asymmetric zero-point.
Fake-quant Linear and Int8Weight: store qweights, dequant on the forward.

NF4 (Dettmers QLoRA): 16-level normal-quantile codebook on last-axis blocks of 64.
NF4Weight / Int8Weight are the numpy API for a frozen quantized Linear base
(QLoRA/QDoRA-style); adapters in other folders wire their own torch copies.

NVFP4-style (NVIDIA 2025):
  x ≈ s_global * s_block * e2m1
  E2M1 levels: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
  block 16 (NVFP4) vs 32 (MXFP4)
  s_block simulated as FP8 E4M3; s_global is FP32.
  Blocks always along the last axis (never ravel).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Int8
# ---------------------------------------------------------------------------


def quant_int8_sym_tensor(x: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    scale = float(np.max(np.abs(x)) / 127.0) if x.size else 1.0
    scale = max(scale, 1e-12)
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale


def dequant_int8_sym_tensor(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * np.float32(scale)


def quant_int8_sym_channel(w: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel symmetric absmax. `axis` is the channel axis (0 = out-features for Linear)."""
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("per-channel quant expects a 2-D weight")
    reduce_axis = 1 if axis == 0 else 0
    absmax = np.max(np.abs(w), axis=reduce_axis, keepdims=True)
    scale = np.maximum(absmax / 127.0, 1e-12)
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32).reshape(-1)


def dequant_int8_sym_channel(q: np.ndarray, scale: np.ndarray, axis: int = 0) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float32)
    if axis == 0:
        return q.astype(np.float32) * scale[:, None]
    return q.astype(np.float32) * scale[None, :]


def quant_int8_sym_row(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LLM.int8 vector-wise row scales for activations X ∈ R^{B×H}."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("row-wise quant expects a 2-D activation")
    absmax = np.max(np.abs(x), axis=1, keepdims=True)
    scale = np.maximum(absmax / 127.0, 1e-12)
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32).reshape(-1)


def dequant_int8_sym_row(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * np.asarray(scale, dtype=np.float32)[:, None]


def quant_int8_asym(x: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Asymmetric uint8 with affine zero-point. x ≈ scale * (q - zp).

    `zp` is the real affine offset `round(-xmin / scale)` and may be negative
    (a shifted-positive tensor needs zp < 0). Clipping zp into [0, 255] would
    collapse [10, 20] onto a single code.
    """
    x = np.asarray(x, dtype=np.float32)
    xmin = float(np.min(x)) if x.size else 0.0
    xmax = float(np.max(x)) if x.size else 0.0
    scale = max((xmax - xmin) / 255.0, 1e-12)
    zp = int(np.round(-xmin / scale))
    q = np.clip(np.round(x / scale + zp), 0, 255).astype(np.uint8)
    return q, scale, zp


def dequant_int8_asym(q: np.ndarray, scale: float, zp: int) -> np.ndarray:
    return np.float32(scale) * (q.astype(np.float32) - np.float32(zp))


class Int8Weight:
    """Per-output-channel int8 storage for Linear W ∈ R^{d_out×d_in} (QLoRA-style base API)."""

    def __init__(self, weight: np.ndarray):
        w = np.asarray(weight, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError("Int8Weight expects a 2-D Linear weight")
        self.qweight, self.scale = quant_int8_sym_channel(w, axis=0)
        self.shape = w.shape

    def dequant(self) -> np.ndarray:
        return dequant_int8_sym_channel(self.qweight, self.scale, axis=0)


class FakeQuantLinear:
    """Linear with quantized weights. Forward dequants then does x @ W^T + b."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None, scheme: str = "sym_tensor"):
        self.scheme = scheme
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        w = np.asarray(weight, dtype=np.float32)
        if scheme == "sym_tensor":
            self.q, self.scale = quant_int8_sym_tensor(w)
            self.zp = None
        elif scheme == "sym_channel":
            self.q, self.scale = quant_int8_sym_channel(w, axis=0)
            self.zp = None
        elif scheme == "asym":
            self.q, self.scale, self.zp = quant_int8_asym(w)
        else:
            raise ValueError(f"unknown scheme {scheme}")

    def weight_fp(self) -> np.ndarray:
        if self.scheme == "sym_tensor":
            return dequant_int8_sym_tensor(self.q, self.scale)
        if self.scheme == "sym_channel":
            return dequant_int8_sym_channel(self.q, self.scale, axis=0)
        return dequant_int8_asym(self.q, self.scale, self.zp)

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float32) @ self.weight_fp().T
        if self.bias is not None:
            y = y + self.bias
        return y


# ---------------------------------------------------------------------------
# NF4 (QLoRA / QDoRA frozen base)
# ---------------------------------------------------------------------------

# bitsandbytes create_normal_map(offset=0.9677083, use_extra_value=True) — QLoRA paper.
NF4_LEVELS = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)


def encode_nf4(x: np.ndarray) -> np.ndarray:
    """Nearest NF4 codebook index in 0..15 (not a float encoding)."""
    x = np.asarray(x, dtype=np.float32)
    return np.argmin(np.abs(x[..., None] - NF4_LEVELS), axis=-1).astype(np.uint8)


def decode_nf4(code: np.ndarray) -> np.ndarray:
    return NF4_LEVELS[np.asarray(code, dtype=np.int32)]


def quantize_nf4(x: np.ndarray, block: int = 64) -> dict:
    """QLoRA NF4: absmax-normalize last-axis blocks, store nearest NormalFloat codes."""
    if block < 1:
        raise ValueError("block must be >= 1")
    blocks, shape, n, pad = _pad_blocks(x, block)
    absmax = np.max(np.abs(blocks), axis=1)
    scale = np.maximum(absmax, 1e-12).astype(np.float32)
    codes = encode_nf4(blocks / scale[:, None])
    return {"codes": codes, "absmax": scale, "shape": shape, "n": n, "block": block, "pad": pad}


def dequant_nf4(pack: dict) -> np.ndarray:
    vals = decode_nf4(pack["codes"]).reshape(-1, pack["block"])
    y = pack["absmax"][:, None] * vals
    last = pack["shape"][-1] if pack["shape"] else pack["n"]
    width = last + int(pack.get("pad", 0))
    y = y.reshape(-1, width)[:, :last]
    return y.reshape(pack["shape"]).astype(np.float32)


class NF4Weight:
    """Frozen NF4 Linear base API (QLoRA/QDoRA-style). Dequant on the fly; codes stay frozen."""

    def __init__(self, weight: np.ndarray, block: int = 64):
        w = np.asarray(weight, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError("NF4Weight expects a 2-D Linear weight")
        self.block = block
        self.pack = quantize_nf4(w, block=block)
        self.shape = w.shape

    def dequant(self) -> np.ndarray:
        return dequant_nf4(self.pack)


# ---------------------------------------------------------------------------
# FP4 E2M1 + hierarchical scales
# ---------------------------------------------------------------------------

# Unsigned magnitudes; sign is stored separately in the 4-bit code (sign << 3 | mag_idx).
E2M1_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def encode_e2m1(x: np.ndarray) -> np.ndarray:
    """Nearest E2M1 code in 0..15 (bit3 = sign, bits0-2 = magnitude index)."""
    x = np.asarray(x, dtype=np.float32)
    sign = (x < 0).astype(np.int32)
    mag = np.abs(x)
    idx = np.argmin(np.abs(mag[..., None] - E2M1_LEVELS), axis=-1).astype(np.int32)
    return ((sign << 3) | idx).astype(np.uint8)


def decode_e2m1(code: np.ndarray) -> np.ndarray:
    code = np.asarray(code, dtype=np.int32)
    sign = np.where((code >> 3) & 1, -1.0, 1.0).astype(np.float32)
    idx = code & 7
    return sign * E2M1_LEVELS[idx]


def quantize_e4m3(x: np.ndarray) -> np.ndarray:
    """Nearest finite E4M3-like value (bias 7, 3-bit mantissa, max 448). Simulated as float."""
    x = np.asarray(x, dtype=np.float32)
    sign = np.sign(x)
    sign = np.where(sign == 0, 1.0, sign)
    ax = np.clip(np.abs(x), 0, 448.0)
    out = np.zeros_like(ax)
    # min positive normal ~ 2^{-6}; below that, round to 0 or subnormal grid 2^{-9}
    tiny = ax > 0
    if not np.any(tiny):
        return out
    ax_t = ax[tiny]
    exp = np.floor(np.log2(np.maximum(ax_t, 2.0 ** -9)))
    exp = np.clip(exp, -9, 8)
    scale = np.power(2.0, exp).astype(np.float32)
    # 3 mantissa bits: 8 bins in [1, 2) for normals; allow [0, 2) when subnormal-ish
    frac = ax_t / scale
    qfrac = np.round(frac * 8.0) / 8.0
    qfrac = np.clip(qfrac, 0.0, 1.875)
    val = np.minimum(qfrac * scale, 448.0)
    out[tiny] = val
    return (sign * out).astype(np.float32)


def _pad_blocks(x: np.ndarray, block: int) -> tuple[np.ndarray, tuple[int, ...], int, int]:
    """Group along the last axis (NVFP4 / NF4 micro-blocks), not ravel order."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 0:
        x = x.reshape(1)
    shape = x.shape
    last = int(shape[-1])
    pad = (block - last % block) % block
    if pad:
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    width = int(x.shape[-1])
    blocks = x.reshape(-1, width).reshape(-1, block)
    return blocks, shape, int(np.prod(shape)), pad


def quantize_nvfp4(x: np.ndarray, block: int = 16) -> dict:
    """Hierarchical NVFP4-style: per-tensor FP32 scale + per-block E4M3 scale + E2M1."""
    if block < 1:
        raise ValueError("block must be >= 1")
    blocks, shape, n, pad = _pad_blocks(x, block)
    absmax = np.max(np.abs(blocks), axis=1)
    s_block_raw = np.maximum(absmax / 6.0, 1e-12)
    s_global = float(max(np.max(s_block_raw) / 448.0, 1e-12))
    s_block = quantize_e4m3(s_block_raw / s_global)
    s_block = np.maximum(s_block, 1e-12)
    scaled = blocks / (s_global * s_block[:, None])
    codes = encode_e2m1(scaled)
    return {"codes": codes, "s_global": s_global, "s_block": s_block, "shape": shape, "n": n, "block": block, "pad": pad}


def dequant_nvfp4(pack: dict) -> np.ndarray:
    vals = decode_e2m1(pack["codes"]).reshape(-1, pack["block"])
    y = pack["s_global"] * pack["s_block"][:, None] * vals
    last = pack["shape"][-1] if pack["shape"] else pack["n"]
    width = last + int(pack.get("pad", 0))
    y = y.reshape(-1, width)[:, :last]
    return y.reshape(pack["shape"]).astype(np.float32)


def quantize_mxfp4(x: np.ndarray, block: int = 32) -> dict:
    """MXFP4-style: E8M0 (power-of-two) scale per block of 32, no second-level scale."""
    blocks, shape, n, pad = _pad_blocks(x, block)
    absmax = np.max(np.abs(blocks), axis=1)
    # E8M0: scale = 2^{round(log2(absmax/6))}
    ratio = np.maximum(absmax / 6.0, 2.0 ** -126)
    s_block = np.power(2.0, np.round(np.log2(ratio))).astype(np.float32)
    s_block = np.maximum(s_block, 1e-30)
    codes = encode_e2m1(blocks / s_block[:, None])
    return {"codes": codes, "s_block": s_block, "shape": shape, "n": n, "block": block, "pad": pad}


def dequant_mxfp4(pack: dict) -> np.ndarray:
    vals = decode_e2m1(pack["codes"]).reshape(-1, pack["block"])
    y = pack["s_block"][:, None] * vals
    last = pack["shape"][-1] if pack["shape"] else pack["n"]
    width = last + int(pack.get("pad", 0))
    y = y.reshape(-1, width)[:, :last]
    return y.reshape(pack["shape"]).astype(np.float32)


def quantize_int4_per_tensor(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Naive symmetric per-tensor int4 (levels -7..7)."""
    x = np.asarray(x, dtype=np.float32)
    scale = float(max(np.max(np.abs(x)) / 7.0, 1e-12)) if x.size else 1.0
    q = np.clip(np.round(x / scale), -7, 7).astype(np.int8)
    return q, scale


def dequant_int4_per_tensor(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * np.float32(scale)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def make_linear_weight(d_out: int, d_in: int, rank: int = 8, seed: int = 0) -> np.ndarray:
    """Structured Linear-like weight: low-rank signal + small noise (not iid noise alone)."""
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(d_out, rank)).astype(np.float32))
    v, _ = np.linalg.qr(rng.normal(size=(d_in, rank)).astype(np.float32))
    s = np.linspace(1.0, 0.05, rank, dtype=np.float32)
    w = (u * s) @ v.T
    w = w + 0.01 * rng.normal(size=w.shape).astype(np.float32)
    return w.astype(np.float32)
