"""Int8 + NF4 + NVFP4 tests — fail if the math is wrong."""

from __future__ import annotations

import numpy as np
import pytest

from quant import (
    E2M1_LEVELS,
    FakeQuantLinear,
    Int8Weight,
    NF4_LEVELS,
    NF4Weight,
    decode_e2m1,
    decode_nf4,
    dequant_int4_per_tensor,
    dequant_int8_asym,
    dequant_int8_sym_channel,
    dequant_int8_sym_row,
    dequant_int8_sym_tensor,
    dequant_mxfp4,
    dequant_nf4,
    dequant_nvfp4,
    encode_e2m1,
    encode_nf4,
    make_linear_weight,
    mse,
    quant_int8_asym,
    quant_int8_sym_channel,
    quant_int8_sym_row,
    quant_int8_sym_tensor,
    quantize_int4_per_tensor,
    quantize_mxfp4,
    quantize_nf4,
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


def test_int8_dequant_uses_scale_not_a_cast():
    """AGENTS: allclose(dequant, W, atol=absmax/127) and not allclose(dequant, qweight.float())."""
    W = make_linear_weight(32, 48, rank=6, seed=11)
    q, scale = quant_int8_sym_tensor(W)
    rec = dequant_int8_sym_tensor(q, scale)
    absmax = float(np.max(np.abs(W)))
    assert np.allclose(rec, W, atol=absmax / 127.0 + 1e-5)
    assert not np.allclose(rec, q.astype(np.float32))
    base = Int8Weight(W)
    assert np.allclose(base.dequant(), dequant_int8_sym_channel(base.qweight, base.scale, axis=0))
    assert not np.allclose(base.dequant(), base.qweight.astype(np.float32))


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


def test_llm_int8_vectorwise_row_scales():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(8, 64)).astype(np.float32)
    x[0, :] *= 40.0  # one hot row must not destroy the others
    q, scale = quant_int8_sym_row(x)
    y = dequant_int8_sym_row(q, scale)
    assert scale.shape == (8,)
    assert scale[0] > 10.0 * scale[1]
    assert np.max(np.abs(x[1] - y[1])) <= scale[1] / 2.0 + 1e-5


def test_e2m1_encode_decode_table():
    levels = np.concatenate([-E2M1_LEVELS[1:][::-1], E2M1_LEVELS])
    codes = encode_e2m1(levels)
    rec = decode_e2m1(codes)
    assert np.allclose(rec, levels, atol=0.0)
    # Midpoints fall to a neighbor (either side is valid); exact levels must stick.
    assert set(E2M1_LEVELS.tolist()) <= set(np.unique(np.abs(rec)).tolist())
    assert decode_e2m1(encode_e2m1(np.array([0.0, 6.0, -6.0, 0.5]))).tolist() == [0.0, 6.0, -6.0, 0.5]


def test_nf4_codebook_and_weight_roundtrip():
    assert NF4_LEVELS.shape == (16,)
    assert NF4_LEVELS[0] == -1.0 and NF4_LEVELS[-1] == 1.0 and 0.0 in NF4_LEVELS
    assert np.allclose(decode_nf4(encode_nf4(NF4_LEVELS)), NF4_LEVELS)
    W = make_linear_weight(24, 80, rank=5, seed=7)  # last % 64 != 0
    pack = quantize_nf4(W, block=64)
    last = W.shape[-1]
    n_blocks = W.shape[0] * ((last + 63) // 64)
    ravel_blocks = (W.size + 63) // 64
    assert n_blocks != ravel_blocks
    assert pack["codes"].shape == (n_blocks, 64)
    assert int(pack["pad"]) == (64 - last % 64) % 64
    rec = dequant_nf4(pack)
    assert rec.shape == W.shape
    assert not np.allclose(rec, W)
    assert float(np.corrcoef(W.ravel(), rec.ravel())[0, 1]) > 0.9
    levels = decode_nf4(pack["codes"])
    assert np.min(np.abs(levels[..., None] - NF4_LEVELS), axis=-1).max() < 1e-6
    consumer = NF4Weight(W, block=64)
    assert np.allclose(consumer.dequant(), rec)


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


def test_nvfp4_last_axis_e2m1_grid_and_oracle():
    """AGENTS / e2e: last%16!=0, E2M1 grid, dequant == s_global*s_block*E2M1, corrcoef, per-block bound."""
    W = make_linear_weight(64, 64, rank=8, seed=3)
    W20 = W[:, :20]
    last = W20.shape[-1]
    assert last % 16 != 0
    n_blocks = W20.shape[0] * ((last + 15) // 16)
    ravel_blocks = (W20.size + 15) // 16
    assert n_blocks != ravel_blocks
    pack = quantize_nvfp4(W20, block=16)
    assert pack["codes"].shape == (n_blocks, 16)
    assert int(pack["pad"]) == (16 - last % 16) % 16
    rec = dequant_nvfp4(pack)
    assert rec.shape == W20.shape
    assert not np.allclose(rec, W20)
    assert float(np.corrcoef(W20.ravel(), rec.ravel())[0, 1]) >= 0.85
    vals = decode_e2m1(pack["codes"]).reshape(-1, 16)
    grid_dist = np.min(np.abs(np.abs(vals)[..., None] - E2M1_LEVELS), axis=-1)
    assert np.all(grid_dist <= 1e-5)
    oracle = (pack["s_global"] * pack["s_block"][:, None] * vals).reshape(W20.shape[0], -1)[:, :last]
    assert np.allclose(rec, oracle, atol=1e-5)
    err = np.abs(W20 - rec)
    for i in range(W20.shape[0]):
        for lo in range(0, last, 16):
            hi = min(lo + 16, last)
            am = float(np.max(np.abs(W20[i, lo:hi])))
            assert float(np.max(err[i, lo:hi])) <= am / 6.0 * 1.5 + 1e-4


def test_dequant_linear_rel_error():
    w = make_linear_weight(32, 64, rank=7, seed=3)
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1.0, size=(8, 64)).astype(np.float32)
    y_fp = x @ w.T
    for scheme in ("sym_tensor", "sym_channel", "asym"):
        layer = FakeQuantLinear(w, scheme=scheme)
        assert np.issubdtype(layer.q.dtype, np.integer)
        assert not np.allclose(layer.weight_fp(), w)
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
