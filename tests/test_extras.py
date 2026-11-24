"""Utilities, crossings, refraction, house positions and the fast position path."""
import os

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
DE440 = os.path.join(HERE, "..", "data", "raw", "de440.bsp")
needs = pytest.mark.skipif(not os.path.exists(DE440), reason="data/raw/de440.bsp not downloaded")


@pytest.fixture(scope="module")
def swe():
    os.environ["ALIF_EPHEMERIS_FILE"] = os.path.abspath(DE440)
    import alif_ephemeris.swe_compat as m
    return m


def test_small_utilities(swe):
    assert swe.csnorm(-100) == 129599900 and swe.d2l(2.5) == 3 and swe.d2l(-2.5) == -3
    assert swe.deg_midp(350, 10) == 0.0 and abs(swe.difrad2n(0.1, 6.2) - 0.1831853) < 1e-6
    assert swe.cs2degstr(12345678) == " 4°17'36" and swe.cs2lonlatstr(12345678, b"N", b"S") == "34N17'37"
    assert swe.cs2timestr(12345678, b":", True) == "10:17:37"
    assert swe.utc_time_zone(2026, 11, 22, 12, 0, 0.0, 5.5) == (2026, 11, 22, 6, 30, 0.0)


def test_refraction_round_trip(swe):
    for h in (1.0, 5.0, 20.0, 60.0):
        app = swe.refrac(h, 1013.25, 15.0, 0)
        assert 0 < app - h < 0.6
        assert abs(swe.refrac(app, 1013.25, 15.0, 1) - h) < 1e-6
    assert swe.refrac(-3.0, 1013.25, 15.0, 0) == -3.0              # below the horizon: unchanged


@needs
def test_crossings_land_on_the_longitude(swe):
    t = swe.solcross_ut(90.0, 2461000.5)
    assert abs(swe.calc_ut(t, swe.SUN, swe.FLG_SWIEPH)[0][0] - 90.0) < 1e-6
    t = swe.mooncross_ut(200.0, 2461000.5)
    assert abs(swe.calc_ut(t, swe.MOON, swe.FLG_SWIEPH)[0][0] - 200.0) < 1e-6
    t, lon, lat = swe.mooncross_node_ut(2461000.5)
    assert abs(lat) < 1e-6
    t = swe.helio_cross_ut(swe.MARS, 100.0, 2461000.5)
    assert abs(swe.calc_ut(t, swe.MARS, swe.FLG_SWIEPH | swe.FLG_HELCTR)[0][0] - 100.0) < 1e-6


def test_house_positions_are_consistent_with_cusps(swe):
    armc, lat, eps = 100.0, 40.0, 23.44
    for system in (b"P", b"K", b"R", b"C", b"B", b"M", b"X", b"E", b"O", b"W"):
        cusps = swe.houses_armc(armc, lat, eps, system)[0]
        for k in (0, 3, 9):
            pos = swe.house_pos(armc, lat, eps, (cusps[k], 0.0), system)
            assert abs((pos - (k + 1) + 6) % 12 - 6) < 0.02, (system, k, pos)


@needs
def test_fast_window_matches_direct_calculation(swe):
    rng = np.random.default_rng(1)
    direct = swe._planet_lonlat_dist
    for body in (swe.SUN, swe.MOON, swe.JUPITER, swe.PLUTO):
        for jd in 2430000.5 + rng.uniform(0, 365.25 * 80, 12):
            swe._WINDOWS.clear()
            whole = float(np.floor(jd))
            fast = swe._fast_lld(body, whole, jd - whole, swe.FLG_SWIEPH, jd)
            slow = direct(body, whole, jd - whole, swe.FLG_SWIEPH, jd)
            assert abs((fast[0] - slow[0] + 180) % 360 - 180) * 3600 < 2e-3
            assert abs(fast[1] - slow[1]) * 3600 < 2e-3


@needs
def test_placidus_refused_inside_polar_circle(swe):
    with pytest.raises(swe.Error):
        swe.houses_ex(2451545.0, 70.0, 20.0, b"P")
    assert len(swe.houses_ex(2451545.0, 70.0, 20.0, b"O")[0]) == 12


@needs
def test_orbital_elements_are_physical(swe):
    jd = 2461000.5 + swe.deltat(2461000.5)
    el = swe.get_orbital_elements(jd, swe.MARS)
    assert abs(el[0] - 1.5237) < 1e-3 and abs(el[1] - 0.0935) < 1e-3 and abs(el[2] - 1.8475) < 2e-3
    assert abs(el[15] - el[0] * (1 - el[1])) < 1e-9 and abs(el[16] - el[0] * (1 + el[1])) < 1e-9
    assert swe.get_orbital_elements(jd, swe.MERCURY)[13] < 0                  # synodic period is signed for inner planets


@needs
def test_nodes_and_apsides_lie_on_the_orbit(swe):
    jd = 2461000.5 + swe.deltat(2461000.5)
    asc, desc, peri, aph = swe.nod_aps(jd, swe.JUPITER, swe.NODBIT_OSCU)
    assert abs(asc[1]) < 0.01 and abs(desc[1]) < 0.01                          # nodes are on the ecliptic
    assert peri[2] < aph[2]
    mx, mn, cur = swe.orbit_max_min_true_distance(jd, swe.JUPITER)
    assert mn < cur < mx


@needs
def test_gauquelin_sector_runs_from_one_to_thirty_seven(swe):
    for method in (0, 1, 2, 4):
        s = swe.gauquelin_sector(2455000.5, swe.MARS, method, (-77.0, 39.0, 0.0))
        assert 1.0 <= s < 37.0
    rise = swe.rise_trans(2455000.5, swe.SUN, swe.CALC_RISE | swe.BIT_DISC_CENTER | swe.BIT_NO_REFRACTION, (-77.0, 39.0, 0.0))[1][0]
    assert abs(swe.gauquelin_sector(rise + 1e-4, swe.SUN, 2, (-77.0, 39.0, 0.0)) - 1.0) < 0.01


@needs
def test_planetocentric_positions(swe):
    jd = 2461000.5 + swe.deltat(2461000.5)
    pos, _ = swe.calc_pctr(jd, swe.MARS, swe.VENUS, swe.FLG_SWIEPH | swe.FLG_SPEED)
    assert 0 <= pos[0] < 360 and 0.2 < pos[2] < 2.7
    # seen from the Earth it must be the ordinary apparent geocentric position
    earth = swe.calc_pctr(jd, swe.MARS, swe.EARTH, swe.FLG_SWIEPH)[0]
    plain = swe.calc(jd, swe.MARS, swe.FLG_SWIEPH)[0]
    assert abs((earth[0] - plain[0] + 180) % 360 - 180) < 3e-4 and abs(earth[2] - plain[2]) < 1e-6


@needs
def test_lunar_eclipse_attributes_and_visibility(swe):
    flag, tret = swe.lun_eclipse_when(2461000.5, swe.FLG_SWIEPH, 0, False)
    how_flag, attr = swe.lun_eclipse_how(tret[0], (-77.0, 39.0, 0.0))
    assert how_flag == swe.ECL_TOTAL and attr[0] > 1.0 and attr[1] > attr[0]
    assert swe.lun_eclipse_how(tret[0] + 0.1, (-77.0, 39.0, 0.0))[0] == 0               # Moon is down then
    bits, times, attrs = swe.lun_eclipse_when_loc(2461000.5, (-77.0, 39.0, 0.0))
    assert bits & swe.ECL_VISIBLE and abs(times[0] - tret[0]) < 1e-3 and times[9] > times[0]
    assert attrs[0] > 1.0


@needs
def test_solar_eclipse_attributes_and_where(swe):
    flag, tret = swe.sol_eclipse_when_glob(2461000.5, swe.FLG_SWIEPH, 0, False)
    kind, geopos, attr = swe.sol_eclipse_where(tret[0])
    lon, lat = geopos[0], geopos[1]
    assert len(geopos) == 10 and len(attr) == 20 and attr[9] > 0 and attr[10] >= 0           # Saros series; the member needs data back to the series start
    assert kind & swe.ECL_CENTRAL and -180 <= lon <= 180 and abs(lat) < 90
    assert 0.9 < attr[0] < 1.1 and 0.9 < attr[1] < 1.1
    how, a = swe.sol_eclipse_how(tret[0], (lon, lat, 0.0))
    assert abs(a[0] - attr[0]) < 0.02                                                    # magnitude agrees at that point
    assert swe.sol_eclipse_how(tret[0], (lon + 180.0, -lat, 0.0))[0] == 0               # far side of the Earth: none


@needs
def test_house_speeds_and_true_horizon(swe):
    cusps, ascmc, cs, asp = swe.houses_ex2(2461000.5, 39.0, -77.0, b"E", 0)
    assert abs(asp[2] - 360.9856) < 0.01 and all(abs(x - asp[0]) < 1e-3 for x in cs)           # equal-house cusps move with the Ascendant
    geo = (-77.0, 39.0, 0.0)
    flat = swe.rise_trans_true_hor(2461000.5, swe.SUN, swe.CALC_RISE, geo, 1013.25, 15.0, 0.0)[1][0]
    hill = swe.rise_trans_true_hor(2461000.5, swe.SUN, swe.CALC_RISE, geo, 1013.25, 15.0, 5.0)[1][0]
    assert 0.01 < hill - flat < 0.08                                                              # a 5 degree hill: later sunrise


def test_new_house_systems_are_well_formed(swe):
    armc, lat, eps = 100.0, 40.0, 23.44
    for letter in (b"L", b"Q", b"U", b"H"):
        cusps, ascmc = swe.houses_armc(armc, lat, eps, letter)
        assert len(cusps) == 12 and all(0 <= c < 360 for c in cusps)
        assert abs((cusps[6] - cusps[0] + 180) % 360 - 180) == pytest.approx(180.0, abs=1e-6)      # cusp 7 opposes cusp 1
    sr = swe.houses_armc(armc, lat, eps, b"L")[0]
    mc = swe.houses_armc(armc, lat, eps, b"L")[1][1]
    q = (swe.houses_armc(armc, lat, eps, b"L")[1][0] - mc) % 360
    assert (sr[10] - mc) % 360 == pytest.approx(q / 4 + 7.5, abs=1e-9)                              # Pullen SD outer house width


def test_gauquelin_sector_houses(swe):
    cusps, ascmc = swe.houses_armc(100.0, 40.0, 23.44, b"G")
    assert len(cusps) == 36 and cusps[0] == pytest.approx(ascmc[0], abs=1e-6)
    with pytest.raises(swe.Error):
        swe.houses_armc(100.0, 70.0, 23.44, b"G")


@needs
def test_new_house_systems_match_reference_when_available(swe):
    real = pytest.importorskip("swisseph")
    for letter in (b"L", b"U", b"H", b"G"):
        for armc, lat in ((100.0, 40.0), (250.0, -33.0), (10.0, 55.0)):
            a, b = swe.houses_armc(armc, lat, 23.44, letter)[0], real.houses_armc(armc, lat, 23.44, letter)[0]
            assert max(abs((x - y + 180) % 360 - 180) for x, y in zip(a, b)) < 0.01, (letter, armc, lat)


@needs
def test_bayer_codes_find_stars(swe):
    from alif_ephemeris import stars
    for code, name in ((",alLeo", "Regulus"), ("alVir", "Spica"), (",al-2Lib", "Zubenelgenubi"), (",beGem", "Pollux")):
        assert stars.find(code).name == name
    assert 0 <= swe.fixstar_ut(",alLeo", 2451545.0, swe.FLG_SWIEPH)[0][0] < 360


def test_refrac_extended_dip_and_round_trip(swe):
    res = swe.refrac_extended(10.0, 100.0, 900.0, 5.0, 0.0065, swe.TRUE_TO_APP)
    assert res[1][3] == pytest.approx(-0.28, abs=0.005) and res[0] > 10.0 and res[1][0] == 10.0
    back = swe.refrac_extended(res[0], 100.0, 900.0, 5.0, 0.0065, swe.APP_TO_TRUE)[0]
    assert back == pytest.approx(10.0, abs=1e-6)
    assert swe.refrac_extended(10.0, 0.0, 1013.25, 15.0, 0.0065, swe.TRUE_TO_APP)[1][3] == 0.0


@needs
def test_lunar_occultations(swe):
    flag, tret = swe.lun_occult_when_glob(2461000.5, "Regulus", swe.FLG_SWIEPH, 0, False)
    assert flag & swe.ECL_TOTAL and tret[2] < tret[0] < tret[3] and tret[0] > 2461000.5
    lon, lat = swe.lun_occult_where(tret[0], "Regulus", swe.FLG_SWIEPH)[1]
    assert -180 <= lon <= 180 and -90 <= lat <= 90
    fl, times, attr = swe.lun_occult_when_loc(2461000.5, 3, (-77.0, 39.0, 0.0))
    assert fl & swe.ECL_VISIBLE and times[1] < times[0] < times[4] and attr[0] == pytest.approx(1.0, abs=1e-6)
    assert attr[7] < 0.3                                                           # centres closer than the Moon's radius
    with pytest.raises(swe.Error):
        swe.lun_occult_when_glob(2461000.5, swe.MOON)


@needs
def test_vulcan_waldemath_and_perigee_exist(swe):
    jd = 2451545.0
    vulcan = swe.calc_ut(jd, swe.VULCAN, swe.FLG_SWIEPH | swe.FLG_SPEED)[0]
    assert 0 <= vulcan[0] < 360 and 0.8 < vulcan[2] < 1.2 and abs(vulcan[1]) < 9         # inside Mercury's orbit, seen from the Earth
    dark = swe.calc_ut(jd, swe.WALDEMATH, swe.FLG_SWIEPH)[0]
    assert 0 <= dark[0] < 360 and 0.004 < dark[2] < 0.01                                  # circles the Earth at about 1 million km
    perigee = swe.calc_ut(jd, swe.INTP_PERG, swe.FLG_SWIEPH)[0]
    apogee = swe.calc_ut(jd, swe.INTP_APOG, swe.FLG_SWIEPH)[0]
    assert perigee[2] < apogee[2] and abs(perigee[1] + apogee[1]) < 3                      # nearer, and the latitudes are opposite


@needs
def test_pullen_ratio_and_sunshine_houses(swe):
    armc, lat, eps = 100.0, 40.0, 23.44
    cusps, ascmc = swe.houses_armc(armc, lat, eps, b"Q")
    asc, mc = ascmc[0], ascmc[1]
    # Pullen sinusoidal ratio: houses 10, 11, 12 are r x, x, r x and houses 7, 8, 9 are r^3 x, r^4 x, r^3 x
    h10, h11, h12 = (cusps[10] - mc) % 360, (cusps[11] - cusps[10]) % 360, (asc - cusps[11]) % 360
    assert h10 == pytest.approx(h12, abs=1e-9)
    q = (asc - mc) % 360
    if q < 90:
        r = h10 / h11
        assert q * r ** 3 * (r + 2) / (2 * r + 1) == pytest.approx(180 - q, abs=1e-6)
    # Sunshine houses: sunrise gives the Ascendant, noon the Midheaven, the cusps keep their order around the zodiac
    cusps, ascmc = swe.houses_armc(armc, lat, eps, b"I", 12.0)                      # the Sun at declination 12 degrees
    assert cusps[0] == pytest.approx(ascmc[0], abs=1e-6) and cusps[9] == pytest.approx(ascmc[1], abs=1e-6)
    spans = [(cusps[(i + 1) % 12] - cusps[i]) % 360 for i in range(12)]
    assert all(0 < x < 180 for x in spans) and sum(spans) == pytest.approx(360.0, abs=1e-6)
    polar_cusps = swe.houses_armc(armc, 80.0, eps, b"I", 20.0)[0]                    # midnight sun: falls back, still 12 cusps
    assert len(polar_cusps) == 12
