"""Time scales, spline reader and the public interface."""
import numpy as np
import pytest

from alif_ephemeris import Ephemeris, SUN, MOON, julian_day, calendar_date, delta_t, delta_t_uncertainty
from alif_ephemeris import timescales as T


def test_julian_day_known_values():
    assert julian_day(2000, 1, 1, 12) == 2451545.0            # J2000.0
    assert abs(julian_day(1957, 10, 4.81) - 2436116.31) < 1e-9  # Meeus example
    assert julian_day(333, 1, 27, 12) == 1842713.0              # Meeus example, Julian calendar
    assert julian_day(1582, 10, 15) - julian_day(1582, 10, 4) == 1.0   # the calendar reform gap


def test_calendar_round_trip():
    for jd in [2451545.0, 1842713.0, 2299160.5, 2299161.5, 0.5, 2460000.123, -1000000.25]:
        y, m, d, h = calendar_date(jd)
        assert abs(julian_day(y, m, d, h) - jd) < 1e-6


def test_delta_t_matches_published_values():
    # Well known values: 1900 about -3 s, 2000 about 63.8 s, 2020 about 69.4 s, 1600 about 109 s
    assert abs(delta_t(julian_day(2000, 1, 1)) - 63.83) < 0.05
    assert abs(delta_t(julian_day(2020, 1, 1)) - 69.36) < 0.05
    assert abs(delta_t(julian_day(1600, 1, 1)) - 109.1) < 0.5
    assert abs(delta_t(julian_day(1900, 1, 1)) + 2.0) < 0.5


def test_delta_t_is_continuous():
    jds = np.linspace(julian_day(1960, 1, 1), julian_day(2060, 1, 1), 5000)
    values = delta_t(jds)
    assert np.max(np.abs(np.diff(values))) < 0.05      # no jumps between neighbouring days


def test_delta_t_uncertainty_grows_with_distance():
    assert delta_t_uncertainty(julian_day(2020, 1, 1)) < 0.01
    assert delta_t_uncertainty(julian_day(2100, 1, 1)) > delta_t_uncertainty(julian_day(2040, 1, 1))
    assert delta_t_uncertainty(julian_day(1000, 1, 1)) > delta_t_uncertainty(julian_day(1800, 1, 1))


def test_tdb_minus_tt_is_small_and_periodic():
    t = np.linspace(2451545, 2451545 + 365.25, 100)
    d = T.tdb_minus_tt(t)
    assert np.max(np.abs(d)) < 0.0018 and np.max(np.abs(d)) > 0.0015


def test_sun_at_j2000(spk):
    eph = Ephemeris(spk.path)
    p = eph.apparent(SUN, jd_ut=julian_day(2000, 1, 1, 12))
    assert abs(p.longitude - 280.3689) < 0.001           # published almanac value
    assert abs(p.distance - 0.98333) < 0.0001
    assert 1.0 < p.longitude_speed < 1.04                 # about a degree a day in early January


def test_vectorised_matches_single(spk):
    eph = Ephemeris(spk.path)
    jds = julian_day(2024, 3, 1) + np.arange(0, 400, 17.3)
    lon, lat, dist = eph.apparent_many(MOON, jds)
    for i in (0, 5, 10, len(jds) - 1):
        one = eph.apparent(MOON, jd_ut=jds[i], speed=False)
        assert abs(one.longitude - lon[i]) < 1e-9


def test_spk_is_thread_safe(spk):
    from concurrent.futures import ThreadPoolExecutor
    eph = Ephemeris(spk.path)
    jd = julian_day(2026, 1, 1, 12)
    ref = eph.apparent(MOON, jd_ut=jd, speed=False).longitude
    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(lambda _: eph.apparent(MOON, jd_ut=jd, speed=False).longitude, range(64)))
    assert all(abs(x - ref) < 1e-12 for x in out)


def test_out_of_range_dates_say_so(spk):
    eph = Ephemeris(spk.path)
    with pytest.raises(ValueError):
        eph.apparent(SUN, jd_ut=julian_day(1500, 1, 1))     # DE440s covers 1849 to 2150


def test_delta_t_array_equals_scalar():
    jds = np.array([julian_day(y, 3, 1) for y in (-800, -300, 500, 1200, 1700, 1850, 1900, 1950, 1972, 1990,
                                                   2010, 2026, 2030, 2045, 2100, 2400, 3000)])
    arr = delta_t(jds)
    for j, a in zip(jds, arr):
        assert abs(delta_t(float(j)) - a) < 1e-12


def test_ephemeris_reports_coverage_and_delta_t_error(spk):
    eph = Ephemeris(spk.path)
    lo, hi = eph.coverage()
    assert julian_day(1849, 12, 1) < lo < julian_day(1850, 1, 5)
    assert julian_day(2150, 1, 1) < hi < julian_day(2150, 2, 1)
    near = eph.apparent(MOON, jd_ut=julian_day(2024, 6, 15, 12))
    far = eph.apparent(MOON, jd_ut=julian_day(2090, 11, 22, 12))
    assert near.delta_t_error_arcsec < 0.01            # measured Delta-T: error negligible
    assert far.delta_t_error_arcsec > 10 * near.delta_t_error_arcsec   # predicted: error grows
    assert eph.apparent(MOON, jd_tt=2461000.5).delta_t_error_arcsec == 0.0


def test_batch_across_two_segments_is_stitched(spk):
    # Build a fake two-segment case by splitting one real segment's time range.
    import copy
    a = copy.copy(spk)
    seg = spk.segments[(399, 3)][0]
    mid = 0.5 * (seg.et_start + seg.et_end)
    from dataclasses import replace
    a.segments = dict(spk.segments)
    a.segments[(399, 3)] = [replace(seg, et_end=mid), replace(seg, et_start=mid)]
    jd = 2451545.0 + np.array([-5000.0, 100.0, 5000.0])
    p1, v1 = spk.relative(399, 3, jd)
    p2, v2 = a.relative(399, 3, jd)
    assert np.allclose(p1, p2, atol=1e-9) and np.allclose(v1, v2, atol=1e-12)
    assert jd[0] < mid / 86400.0 + 2451545.0 < jd[-1]            # the batch really does cross the split
