"""Checks of the core pipeline against independent reference implementations.

Skyfield (MIT) and ERFA (BSD, the IAU SOFA standards) are test oracles only.
Angles are compared with atan2(|cross|, dot), which is accurate for tiny angles.
"""
import numpy as np
import pytest

from alif_ephemeris import frames as F
from alif_ephemeris import planets as P

ASEC = 1 / F.ASEC2RAD
DATES = [2451545.0, 2440000.5, 2460000.5, 2455000.2, 2470000.1, 2445000.7, 2458000.3, 2462000.9,
         2433000.5, 2480000.5]


def angle_mas(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)) * ASEC * 1000


def test_nutation_matches_sofa():
    erfa = pytest.importorskip("erfa")
    for jd in DATES:
        d1, e1 = F.nutation_iau2000a(jd)
        d2, e2 = erfa.nut06a(jd, 0.0)
        assert abs(d1 - d2) * ASEC < 1e-6 and abs(e1 - e2) * ASEC < 1e-6   # under 1 micro-arcsec
        assert np.abs(F.bias_precession_nutation(jd) - erfa.pnm06a(jd, 0.0)).max() * ASEC < 1e-6


def test_positions_match_skyfield(spk):
    erfa = pytest.importorskip("erfa")
    sf = pytest.importorskip("skyfield.api")
    from skyfield.jpllib import SpiceKernel
    k = SpiceKernel(spk.path)
    ts = sf.load.timescale()
    # Barycentre ids (5, 6, 9) are used for the outer planets: Skyfield here has no satellite files, and
    # our planet-centre and newer Uranus and Neptune data are checked against Horizons in test_vs_nasa_horizons.
    names = {P.SUN: "sun", P.MOON: "moon", P.MERCURY: "mercury barycenter",
             P.VENUS: "venus barycenter", P.MARS: "mars barycenter", 5: "jupiter barycenter",
             6: "saturn barycenter", 9: "pluto barycenter"}
    worst = 0.0
    for body, name in names.items():
        for jd in DATES:
            ap = k["earth"].at(ts.tt_jd(jd)).observe(k[name]).apparent().position.au
            theirs = erfa.pnm06a(jd, 0.0) @ ap
            ours = P.apparent_vector_true_equator(spk, body, jd)
            worst = max(worst, angle_mas(ours, theirs))
    assert worst < 0.01, f"worst {worst} mas"      # 0.01 milli-arcsecond = 10 micro-arcsec
