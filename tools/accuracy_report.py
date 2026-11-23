"""Accuracy report: Alif Ephemeris and Swiss Ephemeris, each measured against an independent truth.

Truth = Skyfield (full JPL DE440s, light time, deflection by Sun, Jupiter and Saturn, aberration)
rotated with SOFA's IAU 2006/2000A matrices (ERFA), then to the true ecliptic of date.
Swiss Ephemeris is used here as a black box on a developer machine only, never shipped.

Run:  PYTHONPATH=src python tools/accuracy_report.py [path-to-swiss-ephe-folder]
"""
import os
import sys

import numpy as np

import erfa
from skyfield.api import load
from skyfield.jpllib import SpiceKernel

from alif_ephemeris import Ephemeris, timescales as T
from alif_ephemeris import SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO

DE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "de440s.bsp")
ASEC = 3600.0
# Planet centres and the newest Uranus/Neptune orbits are checked against NASA Horizons in
# tools/horizons_check.py. This report covers the bodies Skyfield can reproduce from DE440s alone.
NAMES = {SUN: "Sun", MOON: "Moon", MERCURY: "Mercury", VENUS: "Venus", MARS: "Mars"}
SKY = {SUN: "sun", MOON: "moon", MERCURY: "mercury barycenter", VENUS: "venus barycenter",
       MARS: "mars barycenter", JUPITER: "jupiter barycenter", SATURN: "saturn barycenter",
       URANUS: "uranus barycenter", NEPTUNE: "neptune barycenter", PLUTO: "pluto barycenter"}
SWE = {SUN: 0, MOON: 1, MERCURY: 2, VENUS: 3, MARS: 4, JUPITER: 5, SATURN: 6, URANUS: 7, NEPTUNE: 8, PLUTO: 9}


def truth(k, ts, body, jd_ut):
    tt = float(T.ut_to_tt(jd_ut))
    ap = k["earth"].at(ts.tt_jd(tt)).observe(k[SKY[body]]).apparent().position.au
    eq = erfa.pnm06a(tt, 0.0) @ ap
    eps = erfa.obl06(tt, 0.0) + erfa.nut06a(tt, 0.0)[1]
    c, s = np.cos(eps), np.sin(eps)
    x, y, z = eq[0], c * eq[1] + s * eq[2], -s * eq[1] + c * eq[2]
    r = np.sqrt(x * x + y * y + z * z)
    return np.degrees(np.arctan2(y, x)) % 360.0, np.degrees(np.arcsin(z / r)), r


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def main():
    import swisseph as swe
    ephe = sys.argv[1] if len(sys.argv) > 1 else None
    if ephe:
        swe.set_ephe_path(ephe)
    eph = Ephemeris(DE)
    k = SpiceKernel(DE)
    ts = load.timescale()
    rng = np.random.default_rng(7)
    dates = T.julian_day(1950, 1, 1) + rng.uniform(0, 365.25 * 100, 60)     # 1950 to 2050
    print("Errors against the independent truth, in arcseconds (worst over %d dates, 1950-2050)" % len(dates))
    print("%-8s %14s %14s %18s" % ("Body", "Alif Ephemeris", "Swiss (same TT)", "Swiss (own delta-T)"))
    dd = [abs(swe.deltat(jd) * 86400.0 - T.delta_t(jd)) for jd in dates]
    print("Delta-T difference Swiss vs ours over these dates: worst %.2f s, median %.2f s" % (max(dd), np.median(dd)))
    for body in NAMES:
        ours = swiss = swiss_ut = 0.0
        for jd in dates:
            tl, tb, td = truth(k, ts, body, jd)
            p = eph.apparent(body, jd_ut=jd, speed=False)
            ours = max(ours, abs(wrap(p.longitude - tl)) * ASEC)
            r, _ = swe.calc(p.jd_tt, SWE[body], swe.FLG_SWIEPH)          # same Terrestrial Time
            swiss = max(swiss, abs(wrap(r[0] - tl)) * ASEC)
            r2, _ = swe.calc_ut(jd, SWE[body], swe.FLG_SWIEPH)          # Swiss does its own delta-T
            swiss_ut = max(swiss_ut, abs(wrap(r2[0] - tl)) * ASEC)
        print("%-8s %14.5f %14.5f %18.5f" % (NAMES[body], ours, swiss, swiss_ut))


if __name__ == "__main__":
    main()
