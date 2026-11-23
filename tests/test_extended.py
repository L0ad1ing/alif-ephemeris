"""Eclipses, asteroids, hypothetical bodies and the Swiss-style interface."""
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
DE440 = os.path.join(HERE, "..", "data", "raw", "de440.bsp")

needs_de440 = pytest.mark.skipif(not os.path.exists(DE440), reason="data/raw/de440.bsp not downloaded")


@pytest.fixture(scope="module")
def swe():
    import alif_ephemeris.swe_compat as compat
    compat.set_ephe_path(os.path.dirname(DE440))
    return compat


def test_interface_calendar_functions_roundtrip(swe):
    assert swe.julday(2000, 1, 1, 12.0) == 2451545.0
    assert swe.revjul(2451545.0) == (2000, 1, 1, 12.0)
    assert swe.GREG_CAL == 1 and swe.FLG_SPEED == 256 and swe.MEAN_NODE == 10


@needs_de440
def test_interface_sun_at_j2000(swe):
    pos, flags = swe.calc_ut(2451545.0, swe.SUN, swe.FLG_SWIEPH | swe.FLG_SPEED)
    assert abs(pos[0] - 280.3689) < 0.002 and 0.95 < pos[3] < 1.05 and abs(pos[2] - 0.98333) < 1e-4


@needs_de440
def test_interface_houses_and_sidereal_time(swe):
    cusps, ascmc = swe.houses_ex(2451545.0, 51.5, -0.1, b"P")
    assert len(cusps) == 12 and len(ascmc) >= 4
    assert abs(((ascmc[1] - cusps[9] + 180) % 360) - 180) < 1e-6          # cusp 10 is the Midheaven in Placidus
    assert abs(swe.sidtime(2451545.0) - 18.69) < 0.01                     # GAST at J2000.0 noon is about 18h 41m


@needs_de440
def test_interface_unsupported_body_says_so(swe):
    with pytest.raises(swe.Error):
        swe.calc_ut(2451545.0, 99999, 0)


@needs_de440
def test_asteroids_exist_and_move_sensibly(swe):
    from alif_ephemeris import smallbodies
    j2000 = {swe.CHIRON: 251.62, swe.CERES: 273.0, swe.VESTA: 273.7}
    chiron = swe.calc_ut(2451545.0, swe.CHIRON, swe.FLG_SWIEPH)[0]
    assert abs(chiron[0] - 251.62) < 0.05                                 # Chiron 11 Sagittarius on 2000-01-01
    assert smallbodies.table_error_km(smallbodies.CHIRON) < 2.0           # fit error at times not used in the fit
    for b in (swe.CERES, swe.PALLAS, swe.JUNO, swe.VESTA, swe.PHOLUS):
        assert 0 <= swe.calc_ut(2451545.0, b, swe.FLG_SWIEPH)[0][0] < 360


@needs_de440
def test_hypothetical_bodies_follow_their_orbits(swe):
    from alif_ephemeris import hypothetical as H
    a = H.ELEMENTS[40][1]
    assert abs(a - 40.99837) < 1e-9                                       # Cupido's published semi-major axis
    r0 = np.linalg.norm(H.heliocentric_ecliptic_j2000(H.OFFSET + 40, 2451545.0))
    assert 40.7 < r0 < 41.3
    lon1 = swe.calc_ut(2451545.0, swe.CUPIDO, swe.FLG_SWIEPH)[0][0]
    lon2 = swe.calc_ut(2451545.0 + 3650, swe.CUPIDO, swe.FLG_SWIEPH)[0][0]
    assert 12.0 < (lon2 - lon1) % 360 < 15.0                              # about 13.7 degrees in ten years (period 262 years)


def _ref():
    return json.load(open(os.path.join(HERE, "data", "eclipse_reference.json")))


@needs_de440
def test_lunar_eclipses_match_reference(swe):
    for r in _ref()["lunar"]:
        kind, t = swe.lun_eclipse_when(r["start"], swe.FLG_SWIEPH, 0, False)
        assert kind == r["kind"]
        assert abs(t[0] - r["tret"][0]) * 86400 < 3.0                      # maximum
        assert abs(t[6] - r["tret"][6]) * 86400 < 4.0 and abs(t[7] - r["tret"][7]) * 86400 < 4.0     # penumbral contacts


@needs_de440
def test_solar_eclipses_match_reference(swe):
    for r in _ref()["solar"]:
        kind, t = swe.sol_eclipse_when_glob(r["start"], swe.FLG_SWIEPH, 0, False)
        assert (kind & 0x7C) == (r["kind"] & 0x7C)                         # total / annular / partial / hybrid
        assert abs(t[0] - r["tret"][0]) * 86400 < 8.0                      # greatest eclipse


@needs_de440
def test_local_eclipses_match_reference(swe):
    for r in _ref()["local"]:
        kind, t, attr = swe.sol_eclipse_when_loc(r["start"], tuple(r["place"]), swe.FLG_SWIEPH, False)
        assert abs(t[0] - r["tret"][0]) * 86400 < 3.0
        assert abs(t[1] - r["tret"][1]) * 86400 < 3.0 and abs(t[4] - r["tret"][4]) * 86400 < 3.0


@needs_de440
def test_sun_rise_set_and_transit_make_sense(swe):
    place = (0.0, 51.5, 0.0)                         # Greenwich, 2000-06-21: a long day
    jd = swe.julday(2000, 6, 21, 0.0)
    _, rise = swe.rise_trans(jd, swe.SUN, swe.CALC_RISE, place)
    _, transit = swe.rise_trans(jd, swe.SUN, swe.CALC_MTRANSIT, place)
    _, sett = swe.rise_trans(jd, swe.SUN, swe.CALC_SET, place)
    hours = lambda t: (t[0] - jd) * 24
    assert 3.5 < hours(rise) < 4.6 and 11.9 < hours(transit) < 12.2 and 19.4 < hours(sett) < 20.6
    assert abs((hours(transit) - hours(rise)) - (hours(sett) - hours(transit))) < 0.05      # symmetric day


@needs_de440
def test_rise_set_match_reference_when_available(swe):
    real = pytest.importorskip("swisseph")
    path = os.environ.get("REFERENCE_EPHE_PATH", "")
    if not os.path.isdir(path):
        pytest.skip("reference data files not found")
    real.set_ephe_path(path)
    for body in (0, 1, 3, "Spica"):
        for place in ((77.2167, 28.6667, 0.0), (-77.0, 39.4, 100.0)):
            for bit in (1, 2, 4):
                a = swe.rise_trans(2451545.0 + 100.3, body, bit, place)
                b = real.rise_trans(2451545.0 + 100.3, body, bit, place)
                assert a[0] == b[0]
                assert abs(a[1][0] - b[1][0]) * 86400 < 8.0


@needs_de440
def test_phenomena_are_sensible(swe):
    venus = swe.pheno(2451545.0, swe.VENUS)
    assert -4.95 < venus[4] < -3.3 and 0.0 <= venus[1] <= 1.0 and 0 <= venus[0] <= 180
    jup = swe.pheno(2451545.0, swe.JUPITER)
    assert -3.2 < jup[4] < -1.5 and 0.001 < jup[3] < 0.02                      # magnitude and disc size in degrees
    moon = swe.pheno(2451545.0, swe.MOON)
    assert np.isnan(moon[4]) and 0.45 < moon[3] < 0.6                           # the Moon's magnitude is not given


@needs_de440
def test_phenomena_match_reference_when_available(swe):
    real = pytest.importorskip("swisseph")
    path = os.environ.get("REFERENCE_EPHE_PATH", "")
    if not os.path.isdir(path):
        pytest.skip("reference data files not found")
    real.set_ephe_path(path)
    for body in (swe.MERCURY, swe.VENUS, swe.MARS, swe.JUPITER, swe.SATURN, swe.NEPTUNE):
        for jd in (2440000.5, 2451545.0, 2458000.5):
            a, b = swe.pheno(jd, body), real.pheno(jd, body, real.FLG_SWIEPH)
            assert abs(a[0] - b[0]) < 0.02 and abs(a[2] - b[2]) < 0.01 and abs(a[4] - b[4]) < 0.08


DE441_1 = os.path.join(HERE, "..", "data", "raw", "de441_part-1.bsp")


@pytest.mark.skipif(not (os.path.exists(DE440) and os.path.exists(DE441_1) and os.path.getsize(DE441_1) > 1_000_000_000),
                    reason="DE440 and DE441 part 1 not downloaded")
def test_two_files_together_cover_ancient_and_modern_dates():
    from alif_ephemeris import Ephemeris, MOON, SUN, julian_day
    eph = Ephemeris([DE440, DE441_1])
    lo, hi = eph.coverage()
    assert lo < julian_day(-9000, 1, 1)                                    # DE441 reaches back past 9000 BC
    modern = eph.apparent(SUN, jd_ut=julian_day(2000, 1, 1, 12), speed=False)
    assert abs(modern.longitude - 280.3689) < 0.002                          # served by DE440
    ancient = eph.apparent(SUN, jd_ut=julian_day(-1000, 3, 21, 12), speed=False)
    assert 0 <= ancient.longitude < 360 and 0.98 < ancient.distance < 1.02   # served by DE441
    both = eph.apparent_many(MOON, [julian_day(-500, 1, 1), julian_day(2000, 1, 1)])   # one batch across both files
    assert len(both[0]) == 2 and np.all(np.isfinite(both[0]))
