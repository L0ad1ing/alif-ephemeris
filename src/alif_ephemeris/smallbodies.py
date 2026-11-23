"""Asteroids and centaurs from compact Chebyshev tables built from NASA JPL Horizons (tools/build_smallbodies.py).

Bodies are numbered 2000000 + the minor-planet number (Ceres 2000001, Chiron 2002060). Tables cover 1800 to 2200.
Each table carries its measured error at times that were not used in the fit.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

AU_KM = 149597870.7
OFFSET = 2000000
NAMES = {1: "Ceres", 2: "Pallas", 3: "Juno", 4: "Vesta", 2060: "Chiron", 5145: "Pholus"}
CERES, PALLAS, JUNO, VESTA, CHIRON, PHOLUS = (OFFSET + n for n in (1, 2, 3, 4, 2060, 5145))


@lru_cache(maxsize=1)
def _tables():
    d = np.load(Path(__file__).with_name("data") / "smallbodies.npz")
    out = {}
    for n in NAMES:
        out[OFFSET + n] = (d["n%d_coef" % n], float(d["n%d_t0" % n]), float(d["n%d_len" % n]), float(d["n%d_err_km" % n]))
    return out


def supports(body) -> bool:
    return body in _tables()


def table_error_km(body) -> float:
    return _tables()[body][3]


def covered(body, jd) -> np.ndarray:
    coef, t0, length, _ = _tables()[body]
    t = (np.asarray(jd, dtype=float) - t0) / length
    return (t >= 0) & (t < len(coef))


def barycentric(body, whole, frac=0.0):
    """Position (km) and velocity (km/s) relative to the solar-system barycentre, ICRF axes. Arrays allowed."""
    coef, t0, length, _ = _tables()[body]
    jd = np.asarray(whole, dtype=float) + np.asarray(frac, dtype=float)
    # (whole - t0) is exact for half-integer whole; adding the fraction last keeps full precision
    t = ((np.asarray(whole, dtype=float) - t0) + np.asarray(frac, dtype=float)) / length
    if np.any(t < 0) or np.any(t >= len(coef)):
        raise ValueError(f"date outside the table for body {body} (it covers 1800 to 2200)")
    k = np.floor(t).astype(int)
    x = (t - k - 0.5) * 2.0
    n = coef.shape[2]
    T = np.empty((n,) + x.shape)
    dT = np.empty_like(T)
    T[0], dT[0] = 1.0, 0.0
    if n > 1:
        T[1], dT[1] = x, 1.0
    for j in range(2, n):
        T[j] = 2 * x * T[j - 1] - T[j - 2]
        dT[j] = 2 * T[j - 1] + 2 * x * dT[j - 1] - dT[j - 2]
    c = coef[k]                                              # (..., 3, n)
    pos = np.einsum("...ak,k...->a...", c, T) * AU_KM
    vel = np.einsum("...ak,k...->a...", c, dT) * (2.0 / length) * AU_KM / 86400.0
    return pos, vel
