"""Ayanamsas, fixed stars, lunar nodes and sidereal positions.

Swiss Ephemeris (pyswisseph) is a development-only numerical reference here; those tests skip without it.
"""
import json
import os

import numpy as np
import pytest

from alif_ephemeris import Ephemeris, MOON, SUN, MARS, ayanamsa as A, julian_day, stars as S, timescales as T

HERE = os.path.dirname(__file__)


def swiss_with_files():
    """pyswisseph pointed at a folder of its data files, or skip. Without files it silently uses a rougher model."""
    swe = pytest.importorskip("swisseph")
    for path in (os.environ.get("ALIF_SWISS_EPHE", ""), os.environ.get("REFERENCE_EPHE_PATH", "")):
        if path and os.path.isdir(path):
            swe.set_ephe_path(path)
            return swe
    pytest.skip("no Swiss Ephemeris data files found (set ALIF_SWISS_EPHE)")


def sep(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0) * 3600.0


def test_lahiri_jhora_is_spica_at_180_and_drifts_at_the_precession_rate():
    """JHora's 'Lahiri' is Spica held at 180 degrees (it matched 518 charts of that program to 0.02 arcsec when we checked)."""
    at = lambda year: float(A.ayanamsa("Lahiri (JHora)", float(T.ut_to_tt(julian_day(year, 1, 1, 12.0)))))
    assert 23.84 < at(2000) < 23.88                                          # the published Lahiri value, 23 deg 51 min
    assert abs((at(2100) - at(2000)) - 1.3969) < 0.005                       # general precession, 50.29 arcsec a year
    star = float(A.ayanamsa("True Chitra (Spica 180)", float(T.ut_to_tt(julian_day(2026, 6, 1, 12.0)))))
    assert abs(star - float(A.ayanamsa("Lahiri (JHora)", float(T.ut_to_tt(julian_day(2026, 6, 1, 12.0)))))) < 1e-9     # the same star, the same number


def test_registry_has_many_systems_and_lookups_work():
    assert len(A.names()) >= 40
    assert A.get("  lahiri ") is A.get("Lahiri")
    with pytest.raises(KeyError):
        A.get("no such ayanamsa")


@pytest.mark.parametrize("name,swiss_id", [
    ("Fagan-Bradley", 0), ("Lahiri", 1), ("DeLuce", 2), ("Raman", 3), ("Krishnamurti (KP)", 5),
    ("Yukteshwar", 7), ("JN Bhasin", 8), ("Suryasiddhanta", 21), ("Lahiri (ICRC)", 46), ("Lahiri 1940", 43),
])
def test_fixed_point_ayanamsas_match_reference_1800_to_2200(name, swiss_id):
    swe = pytest.importorskip("swisseph")
    swe.set_sid_mode(swiss_id, 0, 0)
    for jd in np.linspace(julian_day(1800, 1, 1), julian_day(2200, 1, 1), 12):
        tt = float(T.ut_to_tt(jd))
        ref = swe.get_ayanamsa_ex(tt, swe.FLG_SWIEPH | swe.FLG_NONUT)[1]
        assert sep(float(A.ayanamsa(name, tt)), ref) < 0.1, name


def test_sidereal_planet_is_tropical_minus_true_ayanamsa(spk):
    eph = Ephemeris(spk.path)
    for jd in (julian_day(1975, 3, 3, 4.5), julian_day(2001, 9, 9, 18)):
        tt = float(T.ut_to_tt(jd))
        trop = eph.apparent(MOON, jd_tt=tt, speed=False).longitude
        sid = eph.apparent(MOON, jd_tt=tt, speed=False, sidereal="Lahiri")
        assert sep(sid.longitude, trop - float(A.ayanamsa("Lahiri", tt, true=True))) < 1e-6
        assert abs(sid.ayanamsa - float(A.ayanamsa("Lahiri", tt, true=True))) < 1e-9


def test_sidereal_planet_matches_reference(spk):
    swe = swiss_with_files()
    swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)
    eph = Ephemeris(spk.path)
    for jd, body, sid in ((julian_day(1975, 3, 3, 4.5), SUN, 0), (julian_day(2001, 9, 9, 18), MOON, 1),
                          (julian_day(1950, 6, 6, 0), MARS, 4)):
        tt = float(T.ut_to_tt(jd))
        ref = swe.calc(tt, sid, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)[0][0]
        ours = eph.apparent(body, jd_tt=tt, speed=False, sidereal="Lahiri").longitude
        assert sep(ours, ref) < 0.05


def test_fixed_stars_match_reference():
    swe = swiss_with_files()
    from alif_ephemeris import ephemeris
    import os as _os
    path = _os.path.join(HERE, "..", "data", "raw", "de440s.bsp")
    if not _os.path.exists(path):
        pytest.skip("de440s.bsp missing")
    eph = Ephemeris(path)
    for nm in ("Regulus", "Aldebaran", "Antares", "Algol", "Fomalhaut", "Pollux"):
        for jd in (julian_day(1920, 5, 5), julian_day(2026, 11, 22)):
            tt = float(T.ut_to_tt(jd))
            try:
                ref = swe.fixstar(nm, tt, swe.FLG_SWIEPH)[0]
            except Exception:
                pytest.skip("reference has no star data files")
            lon, lat = S.apparent(eph.spk, S.find(nm), *[float(x) for x in T.split_jd(tt)])
            assert sep(float(lon), ref[0]) < 0.05 and abs(float(lat) - ref[1]) * 3600.0 < 0.05


def test_star_catalogue_basics():
    assert S.find("spica").hip == 65474 and S.find(65474).name == "Spica"
    assert len(S.named_stars()) > 300
    with pytest.raises(KeyError):
        S.find("not a star")


def test_lunar_nodes_match_reference(spk):
    swe = swiss_with_files()
    eph = Ephemeris(spk.path)
    for jd in (julian_day(1955, 7, 7, 1), julian_day(1999, 12, 31, 12), julian_day(2026, 11, 22, 6)):
        tt = float(T.ut_to_tt(jd))
        assert sep(eph.lunar_point("true node", jd), swe.calc(tt, swe.TRUE_NODE, swe.FLG_SWIEPH)[0][0]) < 0.1
        assert sep(eph.lunar_point("mean node", jd), swe.calc(tt, swe.MEAN_NODE, swe.FLG_SWIEPH)[0][0]) < 0.3
        assert sep(eph.lunar_point("osculating apogee", jd), swe.calc(tt, swe.OSCU_APOG, swe.FLG_SWIEPH)[0][0]) < 1.0


def test_mean_node_and_apogee_follow_the_published_polynomials():
    from alif_ephemeris import lunar_points as L
    # Meeus, Astronomical Algorithms ch. 47: the mean node at J2000.0 is 125.0445479 degrees
    assert abs(float(L.mean_node(2451545.0)) - 125.0445479) < 1e-9
    # the apogee sits on the Moon's inclined orbit: its longitude is within 0.12 degrees of the plain polynomial value
    # and its latitude is the tilt of that angle, never more than the 5.145 degree inclination
    lon, lat = L.mean_apogee_lonlat(2451545.0)
    assert abs((float(lon) - (83.3532465 + 180.0) + 180) % 360 - 180) < 0.15 and abs(float(lat)) < 5.1454
    assert abs(float(lat) - 5.145396 * np.sin(np.radians(83.3532465 + 180.0 - 125.0445479))) < 0.05
    # the node regresses by about 19.34 degrees a year
    assert abs((float(L.mean_node(2451545.0 + 365.25)) - float(L.mean_node(2451545.0)) + 180) % 360 - 180 + 19.3414) < 0.001
