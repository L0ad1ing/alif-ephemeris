"""Planet centres: the offset of a planet from the barycentre of its planet-plus-moons system.

The small NASA planet file gives only the system barycentre for Mars to Pluto. The planet's own
centre sits slightly away from it, because the moons pull the barycentre off the planet:
about 1 km for Mars, hundreds of km for Jupiter and Saturn, about 2,100 km for Pluto.

The offsets were refitted from NASA's satellite ephemerides into compact Chebyshev pieces
(tools/build_planet_centres.py). The table covers 1800 to 2200. Each planet's worst refit error
against NASA's data is stored in the file and returned by `refit_error_km`.
Outside the table, the barycentre is used and `covered` says so.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

SECONDS_PER_DAY = 86400.0

BARYCENTRE_OF = {599: 5, 699: 6, 799: 7, 899: 8, 999: 9}
# Newer orbit solutions for Uranus and Neptune (NASA ura184 and nep098, as used by JPL Horizons):
# stored as a small correction to the DE440s barycentre, under the keys 1007 and 1008.
CORRECTED_BARYCENTRES = (7, 8)


@lru_cache(maxsize=1)
def _data():
    d = np.load(Path(__file__).with_name("data") / "planet_centres.npz")
    out = {}
    for body in list(BARYCENTRE_OF) + [1000 + b for b in CORRECTED_BARYCENTRES]:
        out[body] = (d["b%d_coef" % body].astype(np.float64), float(d["b%d_start_jd" % body]),
                     float(d["b%d_len_days" % body]), float(d["b%d_worst_km" % body]))
    return out


def covered(body: int, jd) -> np.ndarray:
    """True where the planet-centre table covers the date(s)."""
    coef, start, length, _ = _data()[body]
    t = (np.asarray(jd, dtype=float) - start) / length
    return (t >= 0) & (t < coef.shape[0])


def refit_error_km(body: int) -> float:
    """Worst error (km) of our compact table against NASA's satellite ephemeris, measured at build time."""
    return _data()[body][3]


def offset(body: int, whole, frac=0.0):
    """A tabulated offset: position (km) and velocity (km/s), ICRF axes.

    `body` 599, 699, 799, 899, 999 gives planet centre minus system barycentre.
    `body` 1007 and 1008 give the newer-solution barycentre minus the DE440s barycentre (Uranus, Neptune).
    Time is `whole + frac` Julian Days (TDB). Dates outside the table give zero.
    """
    coef, start, length, _ = _data()[body]
    whole = np.asarray(whole, dtype=float)
    frac = np.asarray(frac, dtype=float)
    t = ((whole - start) + frac) / length                 # in pieces
    inside = (t >= 0) & (t < coef.shape[0])
    idx = np.clip(np.floor(t).astype(int), 0, coef.shape[0] - 1)
    x = (t - idx - 0.5) * 2.0                             # -1..1 inside a piece
    n = coef.shape[2]
    T = np.empty((n,) + x.shape)
    dT = np.empty_like(T)
    T[0], dT[0] = 1.0, 0.0
    if n > 1:
        T[1], dT[1] = x, 1.0
    for k in range(2, n):
        T[k] = 2 * x * T[k - 1] - T[k - 2]
        dT[k] = 2 * T[k - 1] + 2 * x * dT[k - 1] - dT[k - 2]
    c = coef[idx]                                          # (..., 3, n)
    pos = np.einsum("...ak,k...->a...", c, T)
    vel = np.einsum("...ak,k...->a...", c, dT) * (2.0 / (length * SECONDS_PER_DAY))
    pos = np.where(inside, pos, 0.0)
    vel = np.where(inside, vel, 0.0)
    return pos, vel
