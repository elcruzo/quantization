"""Int8 + FP4 tests — fail if the math is wrong."""

from __future__ import annotations

import numpy as np
import pytest

from quant import (
    E2M1_LEVELS,
    FakeQuantLinear,
    decode_e2m1,
    dequant_int4_per_tensor,
    dequant_int8_asym,
    dequant_int8_sym_tensor,
    dequant_mxfp4,
    dequant_nvfp4,
    encode_e2m1,
    mse,
    quant_int8_asym,
    quant_int8_sym_channel,
    quant_int8_sym_tensor,
    quantize_int4_per_tensor,
    quantize_mxfp4,
    quantize_nvfp4,
)


def test_int8_roundtrip_error_bound():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=2048).astype(np.float32)
    q, scale = quant_int8_sym_tensor(x)
    y = dequant_int8_sym_tensor(q, scale)
    assert np.issubdtype(q.dtype, np.integer)
    # Rounding error is at most half an integer bin.
    assert np.max(np.abs(x - y)) <= scale / 2.0 + 1e-6


def test_zero_point_beats_symmetric_on_shifted():
    rng = np.random.default_rng(1)
    x = rng.uniform(10.0, 20.0, size=1024).astype(np.float32)
    q_s, s_s = quant_int8_sym_tensor(x)
    y_s = dequant_int8_sym_tensor(q_s, s_s)
    q_a, s_a, zp = quant_int8_asym(x)
    y_a = dequant_int8_asym(q_a, s_a, zp)
    assert mse(x, y_a) < mse(x, y_s)
    # Reconstruction should sit inside the original range, not near 0.
    assert abs(float(y_a.mean()) - float(x.mean())) < 0.5


def test_e2m1_encode_decode_table():
    levels = np.concatenate([-E2M1_LEVELS[1:][::-1], E2M1_LEVELS])
    codes = encode_e2m1(levels)
    rec = decode_e2m1(codes)
    assert np.allclose(rec, levels, atol=0.0)
    # Midpoints fall to a neighbor (either side is valid); exact levels must stick.
    assert set(E2M1_LEVELS.tolist()) <= set(np.unique(np.abs(rec)).tolist())
    assert decode_e2m1(encode_e2m1(np.array([0.0, 6.0, -6.0, 0.5]))).tolist() == [0.0, 6.0, -6.0, 0.5]


def test_nvfp4_block16_beats_per_tensor_int4_on_outliers():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 0.08, size=4096).astype(np.float32)
    x[:16] = rng.choice(np.array([-80.0, 80.0], dtype=np.float32), size=16)
    nv = dequant_nvfp4(quantize_nvfp4(x, block=16))
    q4, s4 = quantize_int4_per_tensor(x)
    i4 = dequant_int4_per_tensor(q4, s4)
    mx = dequant_mxfp4(quantize_mxfp4(x, block=32))
    mse_nv, mse_i4, mse_mx = mse(x, nv), mse(x, i4), mse(x, mx)
    assert mse_nv < mse_i4, f"NVFP4 {mse_nv} should beat per-tensor int4 {mse_i4}"
    # Hierarchical block-16 should also beat coarse MX-style block-32 on this tensor.
    assert mse_nv <= mse_mx * 1.05 + 1e-12
    pack = quantize_nvfp4(x, block=16)
    assert pack["codes"].shape[1] == 16
    assert not np.allclose(nv, x)  # lossy, not an identity stub


def test_nvfp4_blocks_along_last_dim():
    x = np.zeros((3, 17), dtype=np.float32)
    x[0, :16] = 1.0
    x[1, 16] = 80.0
    pack = quantize_nvfp4(x, block=16)
    # 17 cols → 2 blocks/row × 3 rows
    assert pack["codes"].shape == (6, 16)
    rec = dequant_nvfp4(pack)
    assert rec.shape == x.shape
    # Row 0's quiet block must not inherit row 1's outlier scale.
    assert rec[0, 0] > 0.5


def test_dequant_linear_rel_error():
    rng = np.random.default_rng(3)
    w = rng.normal(0, 0.3, size=(32, 64)).astype(np.float32)
    x = rng.normal(0, 1.0, size=(8, 64)).astype(np.float32)
    y_fp = x @ w.T
    for scheme in ("sym_tensor", "sym_channel", "asym"):
        layer = FakeQuantLinear(w, scheme=scheme)
        y_q = layer.forward(x)
        rel = np.linalg.norm(y_q - y_fp) / max(np.linalg.norm(y_fp), 1e-12)
        assert rel < 0.08, f"{scheme} rel error {rel}"


def test_per_channel_scale_shape():
    rng = np.random.default_rng(4)
    w = rng.normal(size=(7, 11)).astype(np.float32)
    q, scale = quant_int8_sym_channel(w, axis=0)
    assert q.shape == w.shape
    assert scale.shape == (7,)
    rec = q.astype(np.float32) * scale[:, None]
    assert rec.shape == w.shape
