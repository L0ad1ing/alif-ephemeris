"""Nutation, precession and the ecliptic, following the IAU 2000A / 2006 standards.

References: IERS Conventions 2010 chapter 5; IAU 2006 precession (Capitaine, Wallace, Chapront 2003;
Fukushima-Williams angles); IAU 2000A nutation (Mathews, Herring, Buffett 2002) with the
IAU 2006 adjustment for the Earth's flattening change. The coefficient tables are the
published international standard (see NOTICE).

All functions are pure. Angles in radians, times as Julian Day (TT) floats or numpy arrays.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

ASEC2RAD = 4.848136811095359935899141e-6
J2000 = 2451545.0
_TENTH_USEC_2_RAD = ASEC2RAD * 1e-7  # tenth of a micro-arcsecond = 1e-7 arcsec


@lru_cache(maxsize=1)
def _tables():
    path = Path(__file__).with_name("data") / "iau2000a.npz"
    d = np.load(path)
    return {k: d[k] for k in d}


def _centuries(jd_tt):
    return (np.asarray(jd_tt, dtype=float) - J2000) / 36525.0


_NUT_CACHE: dict = {}


def nutation_iau2000a(jd_tt):
    """Nutation in longitude and obliquity (radians), IAU 2000A with IAU 2006 adjustment."""
    if np.ndim(jd_tt) == 0:                      # one date: remembered, because a position asks for it twice
        key = float(jd_tt)
        hit = _NUT_CACHE.get(key)
        if hit is None:
            if len(_NUT_CACHE) > 4096:
                _NUT_CACHE.clear()
            hit = _NUT_CACHE[key] = _nutation_windowed(key)
        return hit
    arr = np.asarray(jd_tt, dtype=float)
    if arr.size <= 16:                           # a few dates: the fitted windows are far quicker than the full series
        flat = arr.reshape(-1)
        res = [_nutation_windowed(float(x)) for x in flat]
        return (np.array([r[0] for r in res]).reshape(arr.shape), np.array([r[1] for r in res]).reshape(arr.shape))
    return _nutation_uncached(jd_tt)


_NUT_SPAN = 4.0            # days per fitted window; the quickest nutation term has a period of about 5.6 days
_NUT_NODES = 24
_NUT_WINDOWS: dict = {}
_NUT_NODE_X = np.cos(np.pi * (np.arange(_NUT_NODES) + 0.5) / _NUT_NODES)


def _nutation_windowed(jd_tt):
    """Nutation for one date from a Chebyshev fit of the full series over a 4 day window (agrees to 1e-12 rad)."""
    k = int(np.floor((jd_tt - J2000) / _NUT_SPAN))
    win = _NUT_WINDOWS.get(k)
    if win is None:
        if len(_NUT_WINDOWS) > 2048:
            _NUT_WINDOWS.clear()
        centre = J2000 + (k + 0.5) * _NUT_SPAN
        pts = centre + _NUT_NODE_X * (_NUT_SPAN / 2.0)
        dpsi, deps = _nutation_uncached(pts)
        n = _NUT_NODES
        j = np.arange(n)
        basis = np.cos(np.pi * np.outer(j, j + 0.5) / n)            # basis[m, i] = T_m(x_i)
        cpsi = basis @ dpsi * (2.0 / n)
        ceps = basis @ deps * (2.0 / n)
        cpsi[0] /= 2.0
        ceps[0] /= 2.0
        win = _NUT_WINDOWS[k] = (centre, cpsi.tolist(), ceps.tolist())
    centre, cpsi, ceps = win
    x = (jd_tt - centre) / (_NUT_SPAN / 2.0)
    b1p = b2p = b1e = b2e = 0.0
    for m in range(_NUT_NODES - 1, 0, -1):                          # Clenshaw recurrence
        b1p, b2p = 2.0 * x * b1p - b2p + cpsi[m], b1p
        b1e, b2e = 2.0 * x * b1e - b2e + ceps[m], b1e
    return x * b1p - b2p + cpsi[0], x * b1e - b2e + ceps[0]


def _nutation_uncached(jd_tt):
    T = _tables()
    t = _centuries(jd_tt)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)

    fa = T["fa"]  # (5 polynomial orders, 5 arguments), arcseconds
    a = np.zeros((5,) + t.shape)
    for k in range(5):
        a += fa[k][:, None] * t ** k
    a = np.mod(a, 1296000.0) * ASEC2RAD

    arg = T["nals"].astype(float) @ a                     # (678, n)
    s, c = np.sin(arg), np.cos(arg)
    dpsi = T["ls_lon"][:, 0] @ s + (T["ls_lon"][:, 1] @ s) * t + T["ls_lon"][:, 2] @ c
    deps = T["ls_obl"][:, 0] @ c + (T["ls_obl"][:, 1] @ c) * t + T["ls_obl"][:, 2] @ s

    coef, const = T["anomaly_coefficient"], T["anomaly_constant"]
    p = coef[:, None] * t + const[:, None]
    p[-1] *= t                                            # general precession term
    arg = T["napl"].astype(float) @ p                     # (687, n)
    s, c = np.sin(arg), np.cos(arg)
    dpsi = dpsi + T["pl_lon"][:, 0] @ s + T["pl_lon"][:, 1] @ c
    deps = deps + T["pl_obl"][:, 0] @ s + T["pl_obl"][:, 1] @ c

    dpsi *= _TENTH_USEC_2_RAD
    deps *= _TENTH_USEC_2_RAD
    # IAU 2006 adjustment (IERS Conventions 2010, eq. 5.43): small corrections to 2000A
    j2 = -2.7774e-6 * t
    dpsi = dpsi + dpsi * (0.4697e-6 + j2)
    deps = deps + deps * j2
    if scalar:
        return float(dpsi[0]), float(deps[0])
    return dpsi, deps


def mean_obliquity(jd_tt):
    """Mean obliquity of the ecliptic, IAU 2006 (radians)."""
    t = _centuries(jd_tt)
    arcsec = (84381.406 + t * (-46.836769 + t * (-0.0001831 + t * (0.00200340
              + t * (-0.000000576 + t * (-0.0000000434))))))
    return arcsec * ASEC2RAD


def fukushima_williams(jd_tt):
    """The four Fukushima-Williams precession angles (gamma_bar, phi_bar, psi_bar, eps_A), IAU 2006."""
    t = _centuries(jd_tt)
    gamb = (-0.052928 + t * (10.556378 + t * (0.4932044 + t * (-0.00031238
            + t * (-0.000002788 + t * 0.0000000260))))) * ASEC2RAD
    phib = (84381.412819 + t * (-46.811016 + t * (0.0511268 + t * (0.00053289
            + t * (-0.000000440 + t * (-0.0000000176)))))) * ASEC2RAD
    psib = (-0.041775 + t * (5038.481484 + t * (1.5584175 + t * (-0.00018522
            + t * (-0.000026452 + t * (-0.0000000148)))))) * ASEC2RAD
    epsa = mean_obliquity(jd_tt)
    return gamb, phib, psib, epsa


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, s], -1), np.stack([z, -s, c], -1)], -2)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([c, s, z], -1), np.stack([-s, c, z], -1), np.stack([z, z, o], -1)], -2)


def bias_precession_nutation(jd_tt):
    """Matrix taking a GCRS vector to the true equator and equinox of date (IAU 2006/2000A).

    Shape (3, 3) for a scalar time, or (n, 3, 3) for an array.
    """
    gamb, phib, psib, epsa = fukushima_williams(jd_tt)
    dpsi, deps = nutation_iau2000a(jd_tt)
    m = _rot_x(-(epsa + deps)) @ _rot_z(-(psib + dpsi)) @ _rot_x(phib) @ _rot_z(gamb)
    return m


def true_obliquity(jd_tt):
    """True obliquity of date (mean + nutation), radians."""
    _, deps = nutation_iau2000a(jd_tt)
    return mean_obliquity(jd_tt) + deps


def equator_to_ecliptic(vec, obliquity):
    """Rotate vectors from the equator of date to the ecliptic of date. vec shape (..., 3)."""
    c, s = np.cos(obliquity), np.sin(obliquity)
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return np.stack([x, c * y + s * z, -s * y + c * z], -1)


def to_spherical(vec):
    """(longitude 0..2pi, latitude, distance) from x, y, z."""
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    lon = np.mod(np.arctan2(y, x), 2 * np.pi)
    with np.errstate(invalid="ignore", divide="ignore"):          # a zero distance (the Sun seen from the Sun) gives 0
        lat = np.arcsin(np.clip(np.where(r > 0, z / np.where(r > 0, r, 1.0), 0.0), -1.0, 1.0))
    return lon, lat, r
