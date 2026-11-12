from __future__ import annotations

import numpy as np

BANDS = 8
ACF_LAGS = (1, 2, 4, 8)
CORR_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
N_FEATURES = 4 + 4 + 4 * BANDS + 4 * len(ACF_LAGS) + 4 + len(CORR_PAIRS) + 2

def fill_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mask = np.isnan(x)
    if not mask.any():
        return x
    out = np.array(x, dtype=np.float32, copy=True)
    L = out.shape[-1]
    flat = out.reshape(-1, L)
    mflat = mask.reshape(-1, L)
    pos = np.arange(L)
    for r in range(flat.shape[0]):
        mr = mflat[r]
        if not mr.any():
            continue
        v = ~mr
        if v.any():
            flat[r, mr] = np.interp(pos[mr], pos[v], flat[r, v])
        else:
            flat[r] = 0.0
    return out

def features_batch(x: np.ndarray, sample_rate) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m, C, L = x.shape
    nanfrac = np.isnan(x).mean(axis=(1, 2))
    xf = fill_nan(x)
    mu = xf.mean(axis=-1, keepdims=True)
    s = xf.std(axis=-1, keepdims=True) + 1e-9
    z = (xf - mu) / s

    log_s = np.log(s[..., 0])
    ratio = log_s - log_s.mean(axis=1, keepdims=True)
    dc = mu[..., 0] / s[..., 0]

    p = np.abs(np.fft.rfft(z, axis=-1)) ** 2
    p[:, :, 0] = 0.0
    band = np.add.reduceat(p, np.arange(0, L // 2, L // 2 // BANDS), axis=-1)
    shape = band / (band.sum(axis=-1, keepdims=True) + 1e-12)

    acf = np.stack([(z[:, :, :-lag] * z[:, :, lag:]).mean(axis=-1) for lag in ACF_LAGS], axis=-1)
    rough = np.diff(z, axis=-1).std(axis=-1)

    zn = z - z.mean(axis=-1, keepdims=True)
    corr = np.einsum("mct,mdt->mcd", zn, zn) / L
    cc = np.stack([corr[:, i, j] for i, j in CORR_PAIRS], axis=1)

    F = np.concatenate([
        ratio, dc,
        shape.reshape(m, -1),
        acf.reshape(m, -1),
        rough, cc,
        nanfrac[:, None],
        np.asarray(sample_rate, dtype=np.float64).reshape(-1, 1),
    ], axis=1)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def augment(x: np.ndarray, rng: np.random.Generator, strength: float = 1.0,
            gain_lo: float = 0.85, gain_hi: float = 1.18) -> np.ndarray:
    x = np.array(x, dtype=np.float32, copy=True)
    valid = ~np.isnan(x)
    if not valid.any():
        return x
    cnt = valid.sum(axis=-1, keepdims=True)
    mu = np.where(valid, x, 0.0).sum(axis=-1, keepdims=True) / np.maximum(cnt, 1)
    var = np.where(valid, (x - mu) ** 2, 0.0).sum(axis=-1, keepdims=True) / np.maximum(cnt, 1)
    s = np.sqrt(var) + 1e-6

    amp = np.exp(rng.uniform(np.log(0.3), np.log(4.0)))
    if rng.random() < 0.12:
        amp *= rng.uniform(2.0, 5.0)
    x *= amp

    gain = np.exp(rng.uniform(np.log(gain_lo), np.log(gain_hi), size=(x.shape[0], 1)))
    s = s * amp * gain
    x = np.where(valid, x * gain, x)

    if rng.random() < 0.5:
        x = np.where(valid, x + rng.uniform(-1.0, 1.0) * s, x)

    rho = rng.uniform(0.0, 1.0 * strength, size=(x.shape[0], 1))
    alpha = rng.uniform(0.0, 0.6)
    common = rng.standard_normal(x.shape[-1]).astype(np.float32)
    common /= (common.std() + 1e-9)
    noise = np.sqrt(1.0 - alpha) * common[None, :] * (rho * s) \
        + np.sqrt(alpha) * rng.standard_normal(x.shape, dtype=np.float32) * (rho * s)
    x = np.where(valid, x + noise, x)

    if rng.random() < 0.7:
        drift = rng.standard_normal(x.shape[-1]).astype(np.float32)
        drift = np.convolve(drift, np.ones(33, np.float32) / 33.0, mode="same")
        drift /= (drift.std() + 1e-9)
        x = np.where(valid, x + drift * (0.10 * strength * s), x)

    if rng.random() < 0.7:
        target = rng.uniform(0.0, 0.045 * strength)
        L = x.shape[-1]
        for c in range(x.shape[0]):
            missing = np.isnan(x[c]).mean()
            for _ in range(20):
                remaining = int((target - missing) * L)
                if remaining < 2:
                    break
                ln = int(rng.integers(2, min(48, remaining) + 1))
                st = int(rng.integers(0, L - ln + 1))
                x[c, st:st + ln] = np.nan
                missing = np.isnan(x[c]).mean()
    return x
