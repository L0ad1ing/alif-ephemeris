"""Earth rotation angle and sidereal time (IAU 2006 / 2000A, IERS Conventions 2010).

Greenwich mean sidereal time (GMST) from UT1 and TT; Greenwich apparent sidereal time (GAST) adds the
equation of the equinoxes. Local sidereal time adds the east longitude.

Input times are Julian Days. UT1 is taken equal to the UT you give. The difference between UT1 and UTC is
under 0.9 seconds, which is up to 14 arcseconds of sidereal angle; if you need better, pass UT1.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import frames, timescales

ASEC2RAD = frames.ASEC2RAD
TWO_PI = 2.0 * np.pi
J2000 = 2451545.0


def earth_rotation_angle(jd_ut1):
    """Earth rotation angle (radians), IAU 2000. Uses the two-part form to keep full precision."""
    if np.ndim(jd_ut1) == 0:                                  # one date: plain math is several times quicker
        jd = float(jd_ut1)
        whole = math.floor(jd - 0.5) + 0.5
        frac = jd - whole
        t = (whole - J2000) + frac
        f = whole % 1.0 + frac % 1.0
        return (TWO_PI * (f + 0.7790572732640 + 0.00273781191135448 * t)) % TWO_PI
    whole, frac = timescales.split_jd(jd_ut1)
    t = (whole - J2000) + frac
    f = np.mod(whole, 1.0) + np.mod(frac, 1.0)               # fraction of the day
    return np.mod(TWO_PI * (f + 0.7790572732640 + 0.00273781191135448 * t), TWO_PI)


def gmst(jd_ut1, jd_tt=None):
    """Greenwich mean sidereal time in radians (IAU 2006)."""
    if np.ndim(jd_ut1) == 0 and jd_tt is not None and np.ndim(jd_tt) == 0:
        t = (float(jd_tt) - J2000) / 36525.0
        poly = (0.014506 + t * (4612.156534 + t * (1.3915817 + t * (-0.00000044
                + t * (-0.000029956 + t * -0.0000000368))))) * ASEC2RAD
        return (earth_rotation_angle(jd_ut1) + poly) % TWO_PI
    jd_ut1 = np.asarray(jd_ut1, dtype=float)
    if jd_tt is None:
        jd_tt = timescales.ut_to_tt(jd_ut1)
    t = (np.asarray(jd_tt, dtype=float) - J2000) / 36525.0
    poly = (0.014506 + t * (4612.156534 + t * (1.3915817 + t * (-0.00000044
            + t * (-0.000029956 + t * -0.0000000368))))) * ASEC2RAD
    return np.mod(earth_rotation_angle(jd_ut1) + poly, TWO_PI)


@lru_cache(maxsize=1)
def _tables():
    d = np.load(Path(__file__).with_name("data") / "equinox_terms.npz")
    return {k: d[k] for k in d}


def _fundamental(t):
    """The 14 fundamental arguments used by the complementary terms (radians)."""
    t = np.asarray(t, dtype=float)
    tau = TWO_PI

    def poly(c0, c1, c2, c3, c4, turns):
        return ((c0 + t * (c1 + t * (c2 + t * (c3 + t * c4)))) * ASEC2RAD + np.mod(turns * t, 1.0) * tau)

    fa = [
        poly(485868.249036, 715923.2178, 31.8792, 0.051635, -0.00024470, 1325.0),
        poly(1287104.793048, 1292581.0481, -0.5532, 0.000136, -0.00001149, 99.0),
        poly(335779.526232, 295262.8478, -12.7512, -0.001037, 0.00000417, 1342.0),
        poly(1072260.703692, 1105601.2090, -6.3706, 0.006593, -0.00003169, 1236.0),
        poly(450160.398036, -482890.5431, 7.4722, 0.007702, -0.00005939, -5.0),
        4.402608842 + 2608.7903141574 * t,
        3.176146697 + 1021.3285546211 * t,
        1.753470314 + 628.3075849991 * t,
        6.203480913 + 334.0612426700 * t,
        0.599546497 + 52.9690962641 * t,
        0.874016757 + 21.3299104960 * t,
        5.481293872 + 7.4781598567 * t,
        5.311886287 + 3.8133035638 * t,
        (0.024381750 + 0.00000538691 * t) * t,
    ]
    return np.mod(np.stack(np.broadcast_arrays(*fa)), tau)


def _complementary_terms(jd_tt):
    """The complementary terms of the equation of the equinoxes, in arcseconds, for an array of dates (full series)."""
    T = _tables()
    t = (jd_tt - J2000) / 36525.0
    fa = _fundamental(t)                                       # (14, ...)
    arg0 = np.tensordot(T["ke0"].astype(float), fa, axes=(1, 0))      # (33, ...)
    c = np.tensordot(T["se0_sin"], np.sin(arg0), axes=(0, 0)) + np.tensordot(T["se0_cos"], np.cos(arg0), axes=(0, 0))
    arg1 = np.tensordot(T["ke1"].astype(float), fa, axes=(0, 0))
    return c + t * (float(T["se1_sin"]) * np.sin(arg1) + float(T["se1_cos"]) * np.cos(arg1))


_CT_SPAN, _CT_NODES = 4.0, 20
_CT_WINDOWS: dict = {}
_CT_X = np.cos(np.pi * (np.arange(_CT_NODES) + 0.5) / _CT_NODES)
_CT_BASIS = np.cos(np.pi * np.outer(np.arange(_CT_NODES), np.arange(_CT_NODES) + 0.5) / _CT_NODES)


def _complementary_fast(jd_tt):
    """Same value from a Chebyshev fit over a 4 day window (the fastest term has a period of about 9 days)."""
    k = int(np.floor((jd_tt - J2000) / _CT_SPAN))
    win = _CT_WINDOWS.get(k)
    if win is None:
        if len(_CT_WINDOWS) > 2048:
            _CT_WINDOWS.clear()
        centre = J2000 + (k + 0.5) * _CT_SPAN
        vals = _complementary_terms(centre + _CT_X * (_CT_SPAN / 2.0))
        co = _CT_BASIS @ vals * (2.0 / _CT_NODES)
        co[0] /= 2.0
        win = _CT_WINDOWS[k] = (centre, co.tolist())
    centre, co = win
    x = (jd_tt - centre) / (_CT_SPAN / 2.0)
    b1 = b2 = 0.0
    for m in range(_CT_NODES - 1, 0, -1):
        b1, b2 = 2.0 * x * b1 - b2 + co[m], b1
    return x * b1 - b2 + co[0]


def equation_of_equinoxes(jd_tt):
    """Equation of the equinoxes (radians): nutation in longitude times cos(obliquity) plus the
    complementary terms (IERS Conventions 2010, eq. 5.12 and table 5.2e)."""
    jd_tt = np.asarray(jd_tt, dtype=float)
    dpsi, _ = frames.nutation_iau2000a(jd_tt)
    eps = frames.mean_obliquity(jd_tt)
    if jd_tt.size <= 16:
        flat = jd_tt.reshape(-1)
        c = np.array([_complementary_fast(float(x)) for x in flat]).reshape(jd_tt.shape)
        if jd_tt.ndim == 0:
            c = float(c)
    else:
        c = _complementary_terms(jd_tt)
    return dpsi * np.cos(eps) + c * ASEC2RAD


def gast(jd_ut1, jd_tt=None):
    """Greenwich apparent sidereal time in radians."""
    jd_ut1 = np.asarray(jd_ut1, dtype=float)
    if jd_tt is None:
        jd_tt = timescales.ut_to_tt(jd_ut1)
    return np.mod(gmst(jd_ut1, jd_tt) + equation_of_equinoxes(jd_tt), TWO_PI)


def local_apparent_sidereal_time(jd_ut1, east_longitude_deg, jd_tt=None):
    """Local apparent sidereal time in radians for an east longitude in degrees."""
    return np.mod(gast(jd_ut1, jd_tt) + np.radians(east_longitude_deg), TWO_PI)
