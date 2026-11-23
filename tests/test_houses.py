"""House systems and angles, checked against Swiss Ephemeris used as a black-box numerical reference.

pyswisseph is a development-only dependency; these tests skip without it. House cusps need no Swiss data
files. Sidereal time is checked against SOFA (ERFA).
"""
import numpy as np
import pytest

from alif_ephemeris import houses as H, julian_day, sidereal_time as S, timescales as T

swe = pytest.importorskip("swisseph")

SYSTEMS = list("PKORCBTMXEWVDNSF")
RNG = np.random.default_rng(21)
CASES = [(julian_day(1900, 1, 1) + RNG.uniform(0, 365.25 * 150), RNG.uniform(-180, 180), RNG.uniform(-60, 60))
         for _ in range(25)]


def sep(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0) * 3600.0


@pytest.mark.parametrize("letter", SYSTEMS)
def test_house_cusps_match_reference(letter):
    name = H.LETTERS[letter]
    for jd, lon, lat in CASES:
        cusps, ascmc = swe.houses_ex(jd, lat, lon, letter.encode())
        ours = H.houses(jd, lon, lat, name)
        assert ours.system == name
        for k in range(12):
            assert sep(ours.cusps[k], cusps[k]) < 0.02, f"{name} cusp {k + 1} at lat {lat:.1f}"
        assert sep(ours.ascendant, ascmc[0]) < 0.02
        assert sep(ours.midheaven, ascmc[1]) < 0.02
        assert sep(ours.vertex, ascmc[3]) < 0.02


@pytest.mark.parametrize("letter", list("OCRBEWMX"))
def test_polar_latitudes_match_reference(letter):
    rng = np.random.default_rng(5)
    for _ in range(40):
        jd = julian_day(1900, 1, 1) + rng.uniform(0, 365.25 * 150)
        lon = rng.uniform(-180, 180)
        lat = rng.choice([-1, 1]) * rng.uniform(66.0, 89.9)
        cusps, _ = swe.houses_ex(jd, lat, lon, letter.encode())
        ours = H.houses(jd, lon, lat, H.LETTERS[letter])
        for k in range(12):
            assert sep(ours.cusps[k], cusps[k]) < 0.02, f"{H.LETTERS[letter]} cusp {k + 1} at lat {lat:.1f}"


def test_placidus_falls_back_inside_the_polar_circle():
    h = H.houses(julian_day(2000, 1, 1, 12), 10.0, 80.0, "Placidus")
    assert h.system == "Porphyry" and h.notes          # Placidus is undefined here, and we say so


def test_cusps_are_twelve_and_opposite_pairs_hold():
    h = H.houses(julian_day(2026, 11, 22, 12), -77.0, 39.4, "Placidus")
    assert len(h.cusps) == 12
    for k in range(6):
        assert abs((h.cusps[k] - h.cusps[k + 6] + 180.0) % 360.0 - 180.0) in (pytest.approx(180.0, abs=1e-6),)


def test_sidereal_time_matches_sofa():
    erfa = pytest.importorskip("erfa")
    for jd in [julian_day(1900, 5, 5, 3.2), julian_day(2000, 1, 1, 12), julian_day(2026, 11, 22, 7.7)]:
        tt = float(T.ut_to_tt(jd))
        a, b = T.split_jd(jd)
        c, d = T.split_jd(tt)
        sofa = erfa.gst06a(float(a), float(b), float(c), float(d))
        assert abs((S.gast(jd, tt) - sofa + np.pi) % (2 * np.pi) - np.pi) * 206264.806 < 1e-5   # 10 micro-arcsec
