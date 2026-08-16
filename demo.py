"""CPU demo: int8 schemes and FP4 MSE on an outlier-heavy tensor."""

from __future__ import annotations

import numpy as np

from quant import (
    FakeQuantLinear,
    dequant_int4_per_tensor,
    dequant_int8_asym,
    dequant_int8_sym_tensor,
    dequant_mxfp4,
    dequant_nvfp4,
    mse,
    quant_int8_asym,
    quant_int8_sym_tensor,
    quantize_int4_per_tensor,
    quantize_mxfp4,
    quantize_nvfp4,
)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    shifted = rng.uniform(10, 20, size=512).astype(np.float32)
    qs, ss = quant_int8_sym_tensor(shifted)
    qa, sa, zp = quant_int8_asym(shifted)
    print("shifted MSE  sym", mse(shifted, dequant_int8_sym_tensor(qs, ss)))
    print("shifted MSE asym", mse(shifted, dequant_int8_asym(qa, sa, zp)))

    x = rng.normal(0, 0.08, size=4096).astype(np.float32)
    x[:16] = 80.0
    nv = dequant_nvfp4(quantize_nvfp4(x, 16))
    mx = dequant_mxfp4(quantize_mxfp4(x, 32))
    q4, s4 = quantize_int4_per_tensor(x)
    print("outlier MSE  NVFP4", mse(x, nv))
    print("outlier MSE  MXFP4", mse(x, mx))
    print("outlier MSE  int4 ", mse(x, dequant_int4_per_tensor(q4, s4)))

    w = rng.normal(0, 0.3, size=(16, 32)).astype(np.float32)
    xb = rng.normal(size=(4, 32)).astype(np.float32)
    y = xb @ w.T
    yq = FakeQuantLinear(w, scheme="sym_channel").forward(xb)
    print("linear rel", float(np.linalg.norm(yq - y) / np.linalg.norm(y)))
