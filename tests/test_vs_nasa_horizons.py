"""Checks against NASA JPL Horizons answers saved in tests/data/horizons_reference.json.

These are the official JPL results for the Sun, Moon and planets (planet centres, newest Uranus and
Neptune orbits). The test runs offline. Regenerate with tools/make_horizons_fixture.py.
"""
import json
import os

import numpy as np
import pytest

from alif_ephemeris import Ephemeris, planets as P

REF = os.path.join(os.path.dirname(__file__), "data", "horizons_reference.json")
ARCSEC = 206264.806247


def _separation(spk, body, jd, ra, dec):
    whole = float(np.floor(jd - 0.5) + 0.5)
    v = P.astrometric_icrf(spk, body, whole, float(jd - whole))
    o = v / np.linalg.norm(v)
    r, d = np.radians(ra), np.radians(dec)
    t = np.array([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])
    return np.arctan2(np.linalg.norm(np.cross(o, t)), np.dot(o, t)) * ARCSEC


def test_astrometric_places_match_horizons(spk):
    ref = json.load(open(REF))
    for name, b in ref["bodies"].items():
        for jd, ra, dec in zip(ref["jd_tt"], b["ra"], b["dec"]):
            sep = _separation(spk, b["id"], jd, ra, dec)
            if name == "Moon":
                # Horizons uses DE441 for the Moon and we use DE440. They agree to about a metre near
                # 1969 and drift apart smoothly (measured: 0.001 arcsec in 2030, 0.008 in 2145).
                year = 2000 + (jd - 2451545.0) / 365.25
                limit = 0.0002 + 4e-7 * (year - 1969) ** 2
            else:
                limit = 0.0001
            assert sep < limit, f"{name} at JD {jd}: {sep:.6f} arcsec (limit {limit:.6f})"


def test_planet_centre_offsets_are_physical():
    from alif_ephemeris import centres
    whole = np.arange(2459000.5, 2459000.5 + 400, 1.0)
    pos, _ = centres.offset(999, whole)                    # Pluto centre from the Pluto-Charon barycentre
    r = np.linalg.norm(pos, axis=0)
    assert 1900 < r.min() and r.max() < 2400                # about 2,100 km, from Charon's pull
    pj, _ = centres.offset(599, whole)
    rj = np.linalg.norm(pj, axis=0)
    assert 20 < rj.max() < 400                              # tens to hundreds of km for Jupiter


def test_offsets_are_zero_outside_the_table():
    from alif_ephemeris import centres
    pos, vel = centres.offset(599, 2000000.5)               # year 763
    assert np.all(pos == 0) and np.all(vel == 0)
    assert not centres.covered(599, 2000000.5)
    assert centres.covered(599, 2451545.0)


def test_refit_errors_are_small_and_recorded():
    from alif_ephemeris import centres
    for body in (599, 699, 799, 899, 999, 1007, 1008):
        assert centres.refit_error_km(body) < 0.25         # under 250 m, which is under 0.01 mas
