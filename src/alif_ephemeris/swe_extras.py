"""More Swiss-style functions: utilities, crossings, refraction and bookkeeping.

This file is imported at the end of swe_compat and shares its settings. Every function is checked against the
reference program in tests/test_extras.py.
"""
from __future__ import annotations

import functools
import math
import os

import numpy as np
from scipy.optimize import brentq

from . import swe_compat as _s
from . import swe_constants as _c

Error = _s.Error


def _wrap(fn):
    @functools.wraps(fn)
    def inner(*a, **k):
        try:
            return fn(*a, **k)
        except Error:
            raise
        except (ValueError, KeyError, IndexError, RuntimeError, ArithmeticError) as exc:
            raise Error(str(exc)) from exc
    return inner


# ---------------------------------------------------------------------------------------------
# small utilities (angles in degrees, radians or centiseconds)
# ---------------------------------------------------------------------------------------------
CS_DEG = 360000
CS_CIRCLE = 360 * CS_DEG


def csnorm(cs):
    """Centiseconds into 0 to 360 degrees."""
    return int(cs) % CS_CIRCLE


def d2l(x):
    """Round half away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def radnorm(x):
    return x % (2.0 * math.pi)


def difrad2n(p1, p2):
    """p1 - p2 as the shortest signed angle, radians, in -pi to pi."""
    d = (p1 - p2) % (2.0 * math.pi)
    return d - 2.0 * math.pi if d >= math.pi else d


def deg_midp(x1, x2):
    return (x1 + ((x2 - x1 + 180.0) % 360.0 - 180.0) / 2.0) % 360.0


def rad_midp(x1, x2):
    return radnorm(x1 + difrad2n(x2, x1) / 2.0)


def cs2degstr(cs):
    """Degrees within the sign, minutes, seconds: ' 4°17\'36'."""
    cs = csnorm(cs) % (30 * CS_DEG)
    d, rest = divmod(cs, CS_DEG)
    m, rest = divmod(rest, 6000)
    return "%2d°%02d'%02d" % (d, m, rest // 100)


def cs2lonlatstr(cs, pchar, mchar):
    """Signed angle as 34N17'37 (seconds rounded; the seconds are left out when zero)."""
    pchar = pchar.decode() if isinstance(pchar, bytes) else pchar
    mchar = mchar.decode() if isinstance(mchar, bytes) else mchar
    sign = mchar if cs < 0 else pchar
    cs = abs(int(cs))
    cs = (cs + 50) // 100 * 100                     # round to whole seconds
    d, rest = divmod(cs, CS_DEG)
    m, rest = divmod(rest, 6000)
    sec = rest // 100
    if sec == 0:
        return "%d%s%02d" % (d, sign, m)
    return "%d%s%02d'%02d" % (d, sign, m, sec)


def cs2timestr(cs, sep, suppress_zero):
    """Time of day from centiseconds, hh:mm:ss rounded to the second (seconds left out when zero if asked)."""
    sep = sep.decode() if isinstance(sep, bytes) else sep
    cs = (int(cs) + 50) // 100 * 100 if True else int(cs)
    cs %= 24 * 360000
    h, rest = divmod(cs, 360000)
    m, rest = divmod(rest, 6000)
    s = rest // 100
    if s == 0 and suppress_zero:
        return "%02d%s%02d" % (h, sep, m)
    return "%02d%s%02d%s%02d" % (h, sep, m, sep, s)


def utc_time_zone(year, month, day, hour, minute, second, tz_hours):
    """Local time to UTC (positive zone is east of Greenwich). Returns (y, m, d, h, mi, s)."""
    days = _s.julday(year, month, day, 0.0) - 0.5                 # whole day number at midnight
    secs = hour * 3600.0 + minute * 60.0 + second - tz_hours * 3600.0
    shift, secs = divmod(secs, 86400.0)
    y, m, d, _ = _s.revjul(days + shift + 0.5)
    h, rest = divmod(secs, 3600.0)
    mi, sec = divmod(rest, 60.0)
    return (y, m, d, int(h), int(mi), round(sec, 9))


# ---------------------------------------------------------------------------------------------
# refraction
# ---------------------------------------------------------------------------------------------
def _refraction_true(h, pressure, temp):
    """Refraction in degrees for a true altitude h (degrees)."""
    f = (pressure / 1010.0) * (283.0 / (273.0 + temp))
    if h <= 15.0:
        return 1.02 / math.tan(math.radians(h + 10.3 / (h + 5.11))) / 60.0 * f
    t = math.tan(math.radians(90.0 - h))
    return (58.294 * t - 0.0668 * t ** 3) / 3600.0 * f


def refrac(inalt, atpress, attemp, calc_flag=_c.TRUE_TO_APP):
    """True altitude to apparent (calc_flag 0) or the reverse (1), in degrees. No refraction is applied when the
    result would be below the horizon."""
    inalt = float(inalt)
    if calc_flag == _c.TRUE_TO_APP:
        if inalt < -5.0:
            return inalt
        app = inalt + _refraction_true(inalt, atpress, attemp)
        return app if app >= 0.0 else inalt
    # apparent to true: solve true + refraction(true) = apparent
    if inalt < -5.0:
        return inalt
    t = inalt
    for _ in range(20):
        t_new = inalt - _refraction_true(t, atpress, attemp)
        if abs(t_new - t) < 1e-10:
            break
        t = t_new
    return t if t >= 0.0 else inalt


# ---------------------------------------------------------------------------------------------
# crossings: the next time a body reaches a longitude
# ---------------------------------------------------------------------------------------------
def _wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def _find_crossing(fn, target, jd, step, backward=False, limit_days=400.0):
    """First time (searching forward, or backward) at which the angle fn(t) equals target (degrees)."""
    direction = -1.0 if backward else 1.0
    t0 = jd
    v0 = _wrap180(fn(t0) - target)
    t = t0
    while abs(t - jd) < limit_days:
        t1 = t + direction * step
        v1 = _wrap180(fn(t1) - target)
        crossed = (v0 <= 0.0 < v1) or (v0 >= 0.0 > v1)
        if crossed and abs(v1 - v0) < 90.0:
            lo, hi = (t, t1) if t < t1 else (t1, t)
            return float(brentq(lambda x: _wrap180(fn(x) - target), lo, hi, xtol=1e-11))
        t, v0 = t1, v1
    raise Error("no crossing found within %d days" % limit_days)


def _flags_for_cross(flag):
    flag = int(flag)
    return (flag | _c.FLG_SWIEPH) & ~_c.FLG_SPEED


def solcross(x2cross, jd_et, flag=_c.FLG_SWIEPH):
    fl = _flags_for_cross(flag)
    return _find_crossing(lambda t: _s.calc(t, _c.SUN, fl)[0][0], float(x2cross), float(jd_et), 0.5)


def solcross_ut(x2cross, jd_ut, flag=_c.FLG_SWIEPH):
    fl = _flags_for_cross(flag)
    return _find_crossing(lambda t: _s.calc_ut(t, _c.SUN, fl)[0][0], float(x2cross), float(jd_ut), 0.5)


def mooncross(x2cross, jd_et, flag=_c.FLG_SWIEPH):
    fl = _flags_for_cross(flag)
    return _find_crossing(lambda t: _s.calc(t, _c.MOON, fl)[0][0], float(x2cross), float(jd_et), 0.25, limit_days=40.0)


def mooncross_ut(x2cross, jd_ut, flag=_c.FLG_SWIEPH):
    fl = _flags_for_cross(flag)
    return _find_crossing(lambda t: _s.calc_ut(t, _c.MOON, fl)[0][0], float(x2cross), float(jd_ut), 0.25, limit_days=40.0)


def _node_crossing(calc_fn, jd, flag):
    fl = _flags_for_cross(flag)
    lat = lambda t: calc_fn(t, _c.MOON, fl)[0][1]
    step = 0.25
    t = float(jd)
    v0 = lat(t)
    for _ in range(int(40 / step)):
        t1 = t + step
        v1 = lat(t1)
        if (v0 <= 0.0 < v1) or (v0 >= 0.0 > v1):
            tc = float(brentq(lat, t, t1, xtol=1e-11))
            pos = calc_fn(tc, _c.MOON, fl)[0]
            return tc, pos[0], pos[1]
        t, v0 = t1, v1
    raise Error("no node crossing found")


def mooncross_node(jd_et, flag=_c.FLG_SWIEPH):
    return _node_crossing(_s.calc, jd_et, flag)


def mooncross_node_ut(jd_ut, flag=_c.FLG_SWIEPH):
    return _node_crossing(_s.calc_ut, jd_ut, flag)


def helio_cross(planet, x2cross, jd_et, flag=_c.FLG_SWIEPH, dir=0):
    fl = _flags_for_cross(flag) | _c.FLG_HELCTR
    period = {_c.MERCURY: 90.0, _c.VENUS: 230.0, _c.EARTH: 370.0, _c.MARS: 700.0, _c.JUPITER: 4400.0,
              _c.SATURN: 11000.0, _c.URANUS: 31000.0, _c.NEPTUNE: 61000.0, _c.PLUTO: 91000.0}.get(planet, 400.0)
    step = max(period / 200.0, 0.5)
    return _find_crossing(lambda t: _s.calc(t, planet, fl)[0][0], float(x2cross), float(jd_et), step,
                          backward=bool(dir), limit_days=period * 1.2)


def helio_cross_ut(planet, x2cross, jd_ut, flag=_c.FLG_SWIEPH, dir=0):
    dt = _s.deltat(jd_ut)
    return helio_cross(planet, x2cross, float(jd_ut) + dt, flag, dir) - dt


# ---------------------------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------------------------
_TID_ACC = [-25.8]
_LAPSE_RATE = [0.0065]


def deltat_ex(jd, ephe_flag=_c.FLG_SWIEPH):
    return _s.deltat(jd)


def set_tid_acc(t_acc):
    _TID_ACC[0] = float(t_acc)


def get_tid_acc():
    return _TID_ACC[0]


def set_lapse_rate(lapse_rate):
    _LAPSE_RATE[0] = float(lapse_rate)


def set_jpl_file(fname):
    """Use this JPL file as the planetary data file."""
    _s._S.ephe_file = os.fspath(fname) if os.path.isabs(os.fspath(fname)) else _s._S.ephe_file
    _s._DATA_FILES_CACHE.clear()
    _s._RESULT_CACHE.clear()
    _s._WINDOWS.clear()


def get_library_path():
    return os.path.dirname(os.path.abspath(__file__))


def get_current_file_data(ifno=0):
    """(file path, first JD, last JD, JPL number) of the planetary data file in use."""
    files = _s._data_files()
    if ifno >= len(files):
        raise Error("no such file")
    lo, hi = _s._spk().coverage()
    return (files[ifno], float(lo), float(hi), 440 if "440" in os.path.basename(files[ifno]) else 441)


def fixstar_mag(star):
    """Visual magnitude of a star. Returns (magnitude, full star name)."""
    from . import stars
    st = stars.find(star.split(",")[0])
    name = st.name + ("," + st.designation if st.designation else "")
    return (float(st.magnitude), name)


def fixstar2_mag(star):
    return fixstar_mag(star)


for _n in ("refrac", "solcross", "solcross_ut", "mooncross", "mooncross_ut", "mooncross_node", "mooncross_node_ut",
           "helio_cross", "helio_cross_ut", "utc_time_zone", "fixstar_mag", "fixstar2_mag", "get_current_file_data"):
    globals()[_n] = _wrap(globals()[_n])

__all__ = ["csnorm", "d2l", "radnorm", "difrad2n", "deg_midp", "rad_midp", "cs2degstr", "cs2lonlatstr", "cs2timestr",
           "utc_time_zone", "refrac", "solcross", "solcross_ut", "mooncross", "mooncross_ut", "mooncross_node",
           "mooncross_node_ut", "helio_cross", "helio_cross_ut", "deltat_ex", "set_tid_acc", "get_tid_acc",
           "set_lapse_rate", "set_jpl_file", "get_library_path", "get_current_file_data", "fixstar_mag", "fixstar2_mag"]


def house_pos(armc, geolat, eps, objcoord, hsys=b"P"):
    """House position (1.0 to 12.999) of a point (ecliptic longitude, latitude) in the given house system."""
    from . import house_pos as _hp
    name = _s._hsys_name(hsys)
    return _hp.house_position(float(armc), float(geolat), float(eps), float(objcoord[0]), float(objcoord[1]), name)


house_pos = _wrap(house_pos)
__all__.append("house_pos")


def get_orbital_elements(tjd_et, ipl, iflag=_c.FLG_SWIEPH):
    """Osculating Kepler elements (first 17 of 50 slots), Sun-centred, J2000 ecliptic: a, e, i, node, argument of
    perihelion, longitude of perihelion, mean/true/eccentric anomaly, mean longitude, sidereal period (years), mean
    daily motion, tropical period, synodic period (days), time of perihelion, perihelion and aphelion distance."""
    from . import orbits
    pid = _s._PLANET_ID_CENTRES.get(ipl)
    if pid is None or ipl in (_c.SUN, _c.MOON):
        raise Error("orbital elements are not available for body %d" % ipl)
    return tuple(orbits.element_tuple(orbits.elements(_s._spk(), pid, float(tjd_et)))) + (0.0,) * 33


def orbit_max_min_true_distance(tjd_et, ipl, iflag=_c.FLG_SWIEPH):
    """(largest possible distance from the Earth, smallest possible, current), from the orbits' sizes."""
    from . import orbits
    pid = _s._PLANET_ID_CENTRES.get(ipl)
    if pid is None or ipl in (_c.SUN, _c.MOON):
        raise Error("not available for body %d" % ipl)
    return orbits.distance_extremes(_s._spk(), pid, float(tjd_et))


def nod_aps(tjd_et, ipl, method=_c.NODBIT_MEAN, iflag=_c.FLG_SWIEPH):
    """Ascending node, descending node, perihelion and aphelion of a planet's osculating orbit, as geocentric
    positions (longitude, latitude, distance, speeds 0). The mean and osculating methods give the same result here."""
    from . import orbits
    pid = _s._PLANET_ID_CENTRES.get(ipl)
    if pid is None or ipl in (_c.SUN, _c.MOON):
        raise Error("nodes and apsides are not available for body %d" % ipl)
    jd = float(tjd_et)
    el = orbits.elements(_s._spk(), pid, jd)
    earth, _ = orbits.heliocentric_state(_s._spk(), planets_earth(), jd)
    out = []
    p0 = orbits.j2000_ecliptic_to_date(orbits.point_on_orbit(el, 0.0), jd)
    p90 = orbits.j2000_ecliptic_to_date(orbits.point_on_orbit(el, 90.0), jd)
    e_hat = p0 / np.linalg.norm(p0)
    q_hat = np.cross(np.cross(p0, p90), p0)
    q_hat /= np.linalg.norm(q_hat)
    h_hat = np.cross(e_hat, q_hat)                              # orbit normal, in the ecliptic of date
    up = np.array([0.0, 0.0, 1.0])
    node = np.cross(up, h_hat)
    node /= np.linalg.norm(node)
    p_semi = el["a"] * (1 - el["e"] ** 2)

    def at(direction):
        cos_nu = float(direction @ e_hat)
        sin_nu = float(direction @ q_hat)
        r = p_semi / (1 + el["e"] * cos_nu)
        return r * direction

    heliocentric = [at(node), at(-node), el["perihelion_dist"] * e_hat, el["aphelion_dist"] * (-e_hat)]
    earth_date = orbits.j2000_ecliptic_to_date(earth, jd)
    for hv in heliocentric:
        v = hv - earth_date
        r = float(np.linalg.norm(v))
        out.append((float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), float(np.degrees(np.arcsin(v[2] / r))), r, 0.0, 0.0, 0.0))
    return tuple(out)


def nod_aps_ut(tjd_ut, ipl, method=_c.NODBIT_MEAN, iflag=_c.FLG_SWIEPH):
    return nod_aps(float(tjd_ut) + _s.deltat(tjd_ut), ipl, method, iflag)


def planets_earth():
    from . import planets
    return planets.EARTH


for _n in ("get_orbital_elements", "orbit_max_min_true_distance", "nod_aps", "nod_aps_ut"):
    globals()[_n] = _wrap(globals()[_n])
def gauquelin_sector(tjdut, body, method, geopos, atpress=0.0, attemp=0.0, flags=_c.FLG_SWIEPH | _c.FLG_TOPOCTR):
    """Gauquelin sector of a body, 1.0 up to 37.0 (36 sectors; sector 1 starts at rising). Methods 0 and 1 use the
    body's own semi-arcs (with and without its ecliptic latitude); 2 to 5 count time between rising and setting."""
    from . import frames, house_pos as _hp
    jd = float(tjdut)
    if method in (0, 1):
        fl = _c.FLG_SWIEPH | _c.FLG_TOPOCTR
        _s.set_topo(geopos[0], geopos[1], geopos[2] if len(geopos) > 2 else 0.0)
        pos = _s.calc_ut(jd, body, fl)[0] if isinstance(body, int) else _s.fixstar_ut(body, jd, _c.FLG_SWIEPH)[0]
        tt = _s._tt_from_ut(jd)
        eps = float(frames.true_obliquity(tt))
        armc = (float(np.degrees(_s.sidereal_time.local_apparent_sidereal_time(jd, geopos[0], tt))) + _s._st_offset_deg(jd)) % 360.0
        p = _hp.placidus(armc, float(geopos[1]), eps, float(pos[0]), float(pos[1]) if method == 0 else 0.0)
        sector = 38.0 - (3.0 * (p - 1.0) + 1.0)
        return float(sector + 36.0 if sector < 1.0 else sector - 36.0 if sector >= 37.0 else sector)
    bits = _c.BIT_DISC_CENTER if method in (2, 3) else 0
    if method in (2, 4):
        bits |= _c.BIT_NO_REFRACTION
    place = tuple(geopos[:3]) if len(geopos) >= 3 else (geopos[0], geopos[1], 0.0)

    def event(start, kind):
        code, t = _s.rise_trans(start, body, kind | bits, place, atpress, attemp)
        if code != 0:
            raise Error("body does not rise or set here")
        return t[0]
    t0 = jd - 1.1
    rise = event(t0, _c.CALC_RISE)
    for _ in range(4):
        nxt = event(rise + 1e-4, _c.CALC_RISE)
        if nxt > jd:
            break
        rise = nxt
    sett = event(rise + 1e-4, _c.CALC_SET)
    if jd <= sett:
        return 1.0 + 18.0 * (jd - rise) / (sett - rise)
    return 19.0 + 18.0 * (jd - sett) / (nxt - sett)


def _pctr_lonlat(jd_tt, body, centre, flags):
    from . import frames, planets, timescales
    spk = _s._spk()
    w, f = (float(x) for x in timescales.split_jd(jd_tt))
    pc, vc = (np.asarray(a) for a in planets._barycentric(spk, centre, w, f))
    c_km_s = 299792.458
    tau = 0.0
    for _ in range(4):
        pb = np.asarray(planets._barycentric(spk, body, w, f - tau / 86400.0)[0])
        d = pb - pc
        tau = float(np.linalg.norm(d)) / c_km_s
    dist = float(np.linalg.norm(d))
    u, b = d / dist, vc / c_km_s
    g, ub = np.sqrt(1.0 - b @ b), float(d @ b) / dist
    d = (g * u + b * (1.0 + ub / (1.0 + g))) / (1.0 + ub) * dist                  # aberration from the observer's motion
    m = frames.bias_precession_nutation(jd_tt)
    vec = m @ d
    if not flags & _c.FLG_EQUATORIAL:
        vec = frames.equator_to_ecliptic(vec, frames.true_obliquity(jd_tt))
    lon, lat, r = frames.to_spherical(vec)
    lon = float(np.degrees(lon))
    if flags & _c.FLG_SIDEREAL and not flags & _c.FLG_EQUATORIAL:
        lon = (lon - _s._sidereal_ayanamsa(jd_tt, flags)) % 360.0
    return lon, float(np.degrees(lat)), float(r / planets.AU_KM)


def calc_pctr(tjd_et, ipl, iplctr, iflag=_c.FLG_SWIEPH):
    """Position of a planet as seen from another planet (planetocentric): light time and aberration included."""
    body, centre = _s._PLANET_ID_CENTRES.get(ipl), _s._PLANET_ID_CENTRES.get(iplctr)
    if body is None or centre is None or ipl == iplctr:
        raise Error("planetocentric position needs two different planets")
    jd = float(tjd_et)
    base = _pctr_lonlat(jd, body, centre, int(iflag))
    out = [base[0], base[1], base[2], 0.0, 0.0, 0.0]
    if iflag & (_c.FLG_SPEED | _c.FLG_SPEED3):
        a, b = _pctr_lonlat(jd + 0.005, body, centre, int(iflag)), _pctr_lonlat(jd - 0.005, body, centre, int(iflag))
        out[3] = ((a[0] - b[0] + 180.0) % 360.0 - 180.0) / 0.01
        out[4], out[5] = (a[1] - b[1]) / 0.01, (a[2] - b[2]) / 0.01
    return tuple(out), int(iflag)


def _site(geopos):
    return (float(geopos[0]), float(geopos[1]), float(geopos[2]) if len(geopos) > 2 else 0.0)


def _lunar_flag(attrs):
    if attrs["umbral"] >= 1.0:
        return _c.ECL_TOTAL
    if attrs["umbral"] > 0.0:
        return _c.ECL_PARTIAL
    return _c.ECL_PENUMBRAL if attrs["penumbral"] > 0.0 else 0


def lun_eclipse_how(tjd_ut, geopos, flags=_c.FLG_SWIEPH):
    """Attributes of a lunar eclipse at a moment, seen from a place. Returns (type flags, attr): attr[0] umbral
    magnitude, [1] penumbral magnitude, [4] Moon azimuth, [5] true altitude, [6] apparent altitude, [7] distance of
    the Moon from opposition, [8] umbral magnitude. Saros numbers [9] and [10] are 0."""
    from . import eclipse_attrs
    jd = float(tjd_ut)
    a = eclipse_attrs.lunar_attributes(_s._spk(), _s._tt_from_ut(jd), _site(geopos), jd)
    flag = _lunar_flag(a) if a["apparent_altitude"] > 0.0 else 0
    attr = [a["umbral"], a["penumbral"], 0.0, 0.0, a["azimuth"], a["altitude"], a["apparent_altitude"],
            a["opposition_distance"], a["umbral"], 0.0, 0.0] + [0.0] * 9
    return flag, tuple(attr)


def lun_eclipse_when_loc(tjdut, geopos, flags=_c.FLG_SWIEPH, backwards=False):
    """Next (or previous) lunar eclipse that can be seen from a place. Returns (flags, tret, attr): tret[0] maximum,
    [2] and [3] partial begin and end, [4] and [5] total begin and end, [6] and [7] penumbral begin and end (zero
    when the Moon is below the horizon then), [8] moonrise and [9] moonset during the eclipse."""
    from . import eclipse_attrs
    site = _site(geopos)
    start = float(tjdut)
    names = {0: _c.ECL_MAX_VISIBLE, 2: _c.ECL_PARTBEG_VISIBLE, 3: _c.ECL_PARTEND_VISIBLE, 4: _c.ECL_TOTBEG_VISIBLE,
             5: _c.ECL_TOTEND_VISIBLE, 6: _c.ECL_PENUMBBEG_VISIBLE, 7: _c.ECL_PENUMBEND_VISIBLE}
    for _ in range(60):
        flag, tret = _s.lun_eclipse_when(start, flags, 0, bool(backwards))
        times = {k: tret[k] for k in names if tret[k]}

        def up(t):
            return eclipse_attrs._horizon(_s._spk(), _s._PLANET_ID[_c.MOON], _s._tt_from_ut(t), site, t)[2] > 0.0
        visible = {k: up(t) for k, t in times.items()}
        if any(visible.values()):
            out = [0.0] * 10
            out[0] = times[0]
            bits = flag | _c.ECL_VISIBLE
            for k, t in times.items():
                if visible[k]:
                    out[k] = t
                    bits |= names[k]
            lo, hi = min(times.values()), max(times.values())
            for slot, kind in ((8, _c.CALC_RISE), (9, _c.CALC_SET)):
                code, ev = _s.rise_trans(lo - 0.3, _c.MOON, kind, site)
                if code == 0 and lo <= ev[0] <= hi:
                    out[slot] = ev[0]
            return bits, tuple(out), lun_eclipse_how(out[0], site)[1]
        start = tret[0] + (-10.0 if backwards else 10.0)
    raise Error("no visible lunar eclipse found")


def sol_eclipse_how(tjd_ut, geopos, flags=_c.FLG_SWIEPH):
    """Attributes of a solar eclipse at a place and moment: attr[0] fraction of the Sun's diameter covered,
    [1] ratio of the Moon's to the Sun's diameter, [2] fraction of the disc covered, [3] core shadow diameter (km),
    [4] Sun azimuth, [5] true altitude, [6] apparent altitude, [7] separation of centres (degrees), [8] magnitude."""
    from . import eclipse_attrs
    jd = float(tjd_ut)
    a = eclipse_attrs.solar_attributes(_s._spk(), _s._tt_from_ut(jd), _site(geopos), jd)
    if a["magnitude"] <= 0.0 or a["apparent_altitude"] <= 0.0:
        flag = 0
    elif a["magnitude"] >= 1.0:
        flag = (_c.ECL_TOTAL if a["diameter_ratio"] >= 1.0 else _c.ECL_ANNULAR) | _c.ECL_CENTRAL | _c.ECL_VISIBLE
    else:
        flag = _c.ECL_PARTIAL | _c.ECL_NONCENTRAL | _c.ECL_VISIBLE
    attr = [a["magnitude"], a["diameter_ratio"], a["obscuration"], a["core_shadow_km"], a["azimuth"], a["altitude"],
            a["apparent_altitude"], a["separation"], a["magnitude"], 0.0, 0.0] + [0.0] * 9
    return flag, tuple(attr)


def sol_eclipse_where(tjd_ut, flags=_c.FLG_SWIEPH):
    """Where the Moon's shadow axis meets the Earth at a moment. Returns (flags, (longitude, latitude), attr)."""
    from . import eclipse_attrs, eclipses, positions
    jd = float(tjd_ut)
    tt = _s._tt_from_ut(jd)
    rs_v, rm_v = eclipses._sun_moon(_s._spk(), tt)
    a = (rm_v - rs_v) / np.linalg.norm(rm_v - rs_v)
    f = positions.EARTH_FLATTENING
    ell = np.array([1.0, 1.0, 1.0 / (1.0 - f)])
    q, d = rm_v * ell, a * ell                                     # sphere after stretching the polar axis
    b, c = float(q @ d), float(q @ q) - positions.EARTH_RADIUS_KM ** 2
    disc = b * b - float(d @ d) * c
    if disc < 0.0:                                                  # the axis misses the Earth: nearest point to it
        s = -b / float(d @ d)
        p = rm_v + s * a
        p = p / np.linalg.norm(p * ell) * positions.EARTH_RADIUS_KM
        central = 0
    else:
        s = (-b - np.sqrt(disc)) / float(d @ d)
        p = rm_v + s * a
        central = 1
    theta = float(_s.sidereal_time.local_apparent_sidereal_time(jd, 0.0, tt)) + float(np.radians(_s._st_offset_deg(jd)))
    lon = float((np.degrees(np.arctan2(p[1], p[0])) - np.degrees(theta) + 180.0) % 360.0 - 180.0)
    lat = float(np.degrees(np.arctan2(p[2], np.hypot(p[0], p[1]) * (1.0 - f) ** 2)))
    a2 = eclipse_attrs.solar_attributes(_s._spk(), tt, (lon, lat, 0.0), jd)
    attr = [a2["magnitude"], a2["diameter_ratio"], a2["obscuration"], a2["core_shadow_km"], a2["azimuth"], a2["altitude"],
            a2["apparent_altitude"], a2["separation"], a2["magnitude"], 0.0, 0.0] + [0.0] * 9
    if central:
        flag = _c.ECL_TOTAL if a2["diameter_ratio"] >= 1.0 else _c.ECL_ANNULAR
        attr[2] = a2["diameter_ratio"] ** 2
    else:
        flag = _c.ECL_PARTIAL
    attr = tuple(attr)
    return flag | (_c.ECL_CENTRAL if central else _c.ECL_NONCENTRAL), (lon, lat), attr


for _n in ("lun_eclipse_how", "lun_eclipse_when_loc", "sol_eclipse_how", "sol_eclipse_where"):
    globals()[_n] = _wrap(globals()[_n])
__all__ += ["lun_eclipse_how", "lun_eclipse_when_loc", "sol_eclipse_how", "sol_eclipse_where"]

def refrac_extended(inalt, geoalt, atpress, attemp, lapse_rate, calc_flag=_c.TRUE_TO_APP):
    """Refraction for an observer at a height above sea level. Returns (altitude, (true altitude, apparent altitude,
    refraction, dip of the horizon)). The dip uses the terrestrial refraction coefficient for the given pressure,
    temperature and lapse rate; refraction is applied only above the observer's own horizon. The refraction itself
    comes from the same formulas as `refrac`, so it can differ from the reference by up to 0.01 degree."""
    h = max(float(geoalt), 0.0)
    pressure = float(atpress) if atpress else 1013.25
    temp = float(attemp)
    k = 503.0 * pressure / (temp + 273.15) ** 2 * (0.0343 + float(lapse_rate))
    dip = -math.degrees(math.sqrt(2.0 * h / 6371008.8) * math.sqrt(max(0.0, 1.0 - k))) if h > 0.0 else 0.0
    alt = float(inalt)
    if calc_flag == _c.TRUE_TO_APP:
        true_alt = alt
        app = alt + _refraction_true(alt, pressure, temp) if alt > -5.0 else alt
        if app < dip:
            app = alt
    else:
        app = alt
        true_alt = alt
        if alt >= dip and alt > -5.0:
            t = alt
            for _ in range(30):
                t_new = alt - _refraction_true(t, pressure, temp)
                if abs(t_new - t) < 1e-10:
                    break
                t = t_new
            true_alt = t
    result = app if calc_flag == _c.TRUE_TO_APP else true_alt
    return result, (true_alt, app, app - true_alt, dip)


refrac_extended = _wrap(refrac_extended)
__all__.append("refrac_extended")


def _angle_rate(a_plus, a_minus, step):
    return ((a_plus - a_minus + 180.0) % 360.0 - 180.0) / step


def houses_ex2(tjdut, lat, lon, hsys=b"P", flags=0):
    """Like houses_ex, plus the speed of every cusp and angle (degrees per day). Returns
    (cusps, ascmc, cusp speeds, ascmc speeds)."""
    cusps, ascmc = _s.houses_ex(tjdut, lat, lon, hsys, flags)
    h = 1.0 / 1440.0                                                       # one minute either side
    cp, ap = _s.houses_ex(tjdut + h, lat, lon, hsys, flags)
    cm, am = _s.houses_ex(tjdut - h, lat, lon, hsys, flags)
    cs = tuple(_angle_rate(a, b, 2 * h) for a, b in zip(cp, cm))
    asp = tuple(_angle_rate(a, b, 2 * h) for a, b in zip(ap, am))
    return cusps, ascmc, cs, asp


def houses_armc_ex2(armc, lat, eps, hsys=b"P", ascmc9=0.0):
    """Like houses_armc, plus speeds in degrees per day for a sidereal rotation of 360.9856 degrees per day."""
    cusps, ascmc = _s.houses_armc(armc, lat, eps, hsys)
    h = 0.001                                                               # degrees of ARMC
    cp, ap = _s.houses_armc(armc + h, lat, eps, hsys)
    cm, am = _s.houses_armc(armc - h, lat, eps, hsys)
    rate = 360.9856473 / (2 * h)
    cs = tuple(((a - b + 180.0) % 360.0 - 180.0) * rate for a, b in zip(cp, cm))
    asp = [((a - b + 180.0) % 360.0 - 180.0) * rate for a, b in zip(ap, am)]
    asp[2] = 360.9856473
    return cusps, ascmc, cs, tuple(asp)


def rise_trans_true_hor(tjdut, body, rsmi, geopos, atpress=0.0, attemp=0.0, horhgt=0.0, flags=_c.FLG_SWIEPH):
    """Like rise_trans, with the visible horizon at a height of `horhgt` degrees instead of 0."""
    from . import risetrans
    original = risetrans.depression_for_apparent
    press = float(atpress) if atpress else 1013.25
    temp = float(attemp) if atpress else 15.0

    def shifted(*a, **k):
        # the horizon refraction is replaced by the refraction at the height of the horizon
        if horhgt == 0.0:
            refr_here = 34.0 / 60.0 * (press / 1013.25) * (288.15 / (273.15 + temp))      # the standard 34 arcmin
        else:
            refr_here = _refraction_true(float(horhgt), press, temp) if horhgt > -0.5 else 0.0
        return original(*a, **k) + float(horhgt) + risetrans.horizon_refraction(press, temp) - refr_here
    risetrans.depression_for_apparent = shifted
    try:
        return _s.rise_trans(tjdut, body, rsmi, geopos, atpress, attemp, flags)
    finally:
        risetrans.depression_for_apparent = original


for _n in ("houses_ex2", "houses_armc_ex2", "rise_trans_true_hor"):
    globals()[_n] = _wrap(globals()[_n])
__all__ += ["houses_ex2", "houses_armc_ex2", "rise_trans_true_hor"]

def _occ_target(body):
    from . import occultations, stars
    if isinstance(body, str):
        try:
            return occultations.Target(stars.find(_s._star_key(body)))
        except KeyError:
            raise Error(f"star {body!r} not found")
    pid = _s._PLANET_ID_CENTRES.get(body)
    if pid is None or body in (_c.MOON, _c.EARTH):
        raise Error("the Moon can only occult a planet or a star")
    return occultations.Target(pid)


def lun_occult_when_glob(tjdut, body, flags=_c.FLG_SWIEPH, ecltype=0, backwards=False):
    """Next (or previous) occultation of a planet or star by the Moon visible anywhere on Earth. Returns
    (flags, tret): tret[0] maximum, [2] and [3] begin and end of the shadow on Earth, [4] and [5] begin and end of
    the full-width shadow, [6] and [7] begin and end of the central line (UT)."""
    from . import occultations
    target = _occ_target(body)
    occ = occultations.next_global_occultation(_s._spk(), target, _s._tt_from_ut(tjdut), bool(backwards))
    if occ is None:
        raise Error("no occultation found")
    ut = _s._ut_from_tt
    t = [0.0] * 10
    t[0] = ut(occ.maximum)
    t[2], t[3] = ut(occ.partial[0]), ut(occ.partial[1])
    t[4], t[5] = ut(occ.umbral[0]), ut(occ.umbral[1])
    if occ.central:
        t[6], t[7] = ut(occ.centre_line[0]), ut(occ.centre_line[1])
    return _c.ECL_TOTAL | (_c.ECL_CENTRAL if occ.central else _c.ECL_NONCENTRAL), tuple(t)


def lun_occult_when_loc(tjdut, body, geopos, flags=_c.FLG_SWIEPH, backwards=False):
    """Next (or previous) occultation seen from a place. Returns (flags, tret, attr): tret[0] maximum and [1] to [4]
    the four contacts (second and third are the same as first and fourth when the target is a point)."""
    from . import eclipse_attrs, occultations
    target = _occ_target(body)
    site = _site(geopos)
    jd = float(tjdut)
    moon = _s._PLANET_ID[_c.MOON]
    for _attempt in range(400):                                   # skip occultations that happen with the Moon below the horizon
        occ = occultations.next_local_occultation(_s._spk(), target, _s._tt_from_ut(jd), site, _s.deltat(jd), bool(backwards))
        if occ is None:
            raise Error("no occultation found")
        contacts = [occ.maximum, occ.first, occ.fourth] + [x for x in (occ.second, occ.third) if x is not None]
        if any(eclipse_attrs._horizon(_s._spk(), moon, c, site, _s._ut_from_tt(c))[2] > 0.0 for c in contacts):
            break
        jd = _s._ut_from_tt(occ.maximum) + (-1.0 if backwards else 1.0)
    else:
        raise Error("no visible occultation found")
    ut = _s._ut_from_tt
    t = [0.0] * 10
    t[0], t[1], t[4] = ut(occ.maximum), ut(occ.first), ut(occ.fourth)
    if target.is_star:
        t[2], t[3] = t[1], t[4]
    elif occ.second is not None:
        t[2], t[3] = ut(occ.second), ut(occ.third)
    sun = _s._PLANET_ID[_c.SUN]
    flag = (_c.ECL_TOTAL if (occ.second is not None or target.is_star) else _c.ECL_PARTIAL) | _c.ECL_VISIBLE
    names = {0: _c.ECL_MAX_VISIBLE, 1: _c.ECL_1ST_VISIBLE, 2: _c.ECL_2ND_VISIBLE, 3: _c.ECL_3RD_VISIBLE, 4: _c.ECL_4TH_VISIBLE}
    any_visible = False
    for k, bit in names.items():
        if t[k] and eclipse_attrs._horizon(_s._spk(), moon, _s._tt_from_ut(t[k]), site, t[k])[2] > 0.0:
            flag |= bit
            any_visible = True
    if not any_visible:
        flag &= ~_c.ECL_VISIBLE
    for k, bit in ((1, _c.ECL_OCC_BEG_DAYLIGHT), (4, _c.ECL_OCC_END_DAYLIGHT)):
        if eclipse_attrs._horizon(_s._spk(), sun, _s._tt_from_ut(t[k]), site, t[k])[2] > 0.0:
            flag |= bit
    sep, r_m, r_t = occultations._local_separation(_s._spk(), target, occ.maximum, site, t[0])
    az, alt, app = eclipse_attrs._horizon(_s._spk(), moon, occ.maximum, site, t[0])
    ratio = r_m / r_t if r_t > 0 else 0.0
    cover = 1.0 if (target.is_star or occ.second is not None) else max(0.0, min(1.0, eclipse_attrs._overlap_fraction(r_t, r_m, sep)))
    attr = [min(1.0, occ.magnitude), ratio, cover, -2.0 * occultations.eclipses.R_MOON_SOLAR, az, alt, app,
            float(np.degrees(sep))] + [0.0] * 12
    return flag, tuple(t), tuple(attr)


def lun_occult_where(tjdut, body, flags=_c.FLG_SWIEPH):
    """Where the Moon's shadow axis meets the Earth at a moment, for an occultation. Returns (flags, (lon, lat), attr)."""
    from . import eclipse_attrs, occultations, positions
    target = _occ_target(body)
    jd = float(tjdut)
    tt = _s._tt_from_ut(jd)
    rm = occultations._moon(_s._spk(), tt)
    rt = target.vector(_s._spk(), tt)
    a = (rm - rt) / np.linalg.norm(rm - rt)
    f = positions.EARTH_FLATTENING
    ell = np.array([1.0, 1.0, 1.0 / (1.0 - f)])
    q, d = rm * ell, a * ell
    b, c = float(q @ d), float(q @ q) - positions.EARTH_RADIUS_KM ** 2
    disc = b * b - float(d @ d) * c
    central = disc >= 0.0
    if central:
        s_par = (-b - np.sqrt(disc)) / float(d @ d)
        p = rm + s_par * a
    else:
        s_par = -b / float(d @ d)
        p = rm + s_par * a
        p = p / np.linalg.norm(p * ell) * positions.EARTH_RADIUS_KM
    theta = float(_s.sidereal_time.local_apparent_sidereal_time(jd, 0.0, tt)) + float(np.radians(_s._st_offset_deg(jd)))
    lon = float((np.degrees(np.arctan2(p[1], p[0])) - np.degrees(theta) + 180.0) % 360.0 - 180.0)
    lat = float(np.degrees(np.arctan2(p[2], np.hypot(p[0], p[1]) * (1.0 - f) ** 2)))
    site = (lon, lat, 0.0)
    sep, r_m, r_t = occultations._local_separation(_s._spk(), target, tt, site, jd)
    moon = _s._PLANET_ID[_c.MOON]
    az, alt, app = eclipse_attrs._horizon(_s._spk(), moon, tt, site, jd)
    ratio = r_m / r_t if r_t > 0 else 0.0
    mag = 1.0 if (target.is_star or sep < r_m - r_t) else max(0.0, (r_m + r_t - sep) / (2 * r_t))
    attr = [mag, ratio, 1.0 if mag >= 1.0 else mag, -2.0 * occultations.eclipses.R_MOON_SOLAR, az, alt, app, float(np.degrees(sep))]
    flag = _c.ECL_TOTAL | (_c.ECL_CENTRAL if central else _c.ECL_NONCENTRAL)
    return flag, (lon, lat), tuple(attr)


for _n in ("lun_occult_when_glob", "lun_occult_when_loc", "lun_occult_where"):
    globals()[_n] = _wrap(globals()[_n])
__all__ += ["lun_occult_when_glob", "lun_occult_when_loc", "lun_occult_where"]

# ---------------------------------------------------------------------------------------------
# heliacal events (classical "arcus visionis" model) and the limiting magnitude
# ---------------------------------------------------------------------------------------------
HELIACAL_RISING, HELIACAL_SETTING = 1, 2
MORNING_FIRST, EVENING_LAST, EVENING_FIRST, MORNING_LAST = 1, 2, 3, 4
ACRONYCHAL_RISING, ACRONYCHAL_SETTING, COSMICAL_SETTING = 5, 6, 6
AUNIT_TO_KM, AUNIT_TO_LIGHTYEAR, AUNIT_TO_PARSEC = 149597870.7, 1.5812507409819728e-05, 4.848136811095274e-06
PHOTOPIC_FLAG, SCOTOPIC_FLAG, MIXEDOPIC_FLAG = 0, 1, 2
TIDAL_AUTOMATIC, TIDAL_DEFAULT = 999999, -25.8
TIDAL_DE200, TIDAL_DE403, TIDAL_DE404, TIDAL_DE405, TIDAL_DE406 = -23.8946, -25.580, -25.580, -25.826, -25.826
TIDAL_DE421, TIDAL_DE430, TIDAL_DE431, TIDAL_DE441 = -25.85, -25.82, -25.80, -25.936
TIDAL_26 = -26.0

# Ptolemy's arcus visionis (Almagest and Phaseis): how far below the horizon the Sun must be for the object to be seen
_ARCUS_PLANETS = {"mercury": 10.0, "venus": 5.0, "mars": 14.5, "jupiter": 10.0, "saturn": 11.0, "uranus": 12.0, "neptune": 13.0,
                  "pluto": 17.0}
_PLANET_BY_NAME = {"sun": _c.SUN, "moon": _c.MOON, "mercury": _c.MERCURY, "venus": _c.VENUS, "mars": _c.MARS, "jupiter": _c.JUPITER,
                   "saturn": _c.SATURN, "uranus": _c.URANUS, "neptune": _c.NEPTUNE, "pluto": _c.PLUTO}


def _arcus_visionis(name, magnitude):
    key = name.strip().lower()
    if key in _ARCUS_PLANETS:
        return _ARCUS_PLANETS[key]
    for limit, arc in ((1.5, 12.0), (2.5, 13.0), (3.5, 14.0), (4.5, 15.0), (5.5, 16.0)):
        if magnitude < limit:
            return arc
    return 17.0


def _object_magnitude(name, jd_ut):
    key = name.strip().lower()
    if key in _PLANET_BY_NAME and key not in ("sun", "moon"):
        return _s.pheno_ut(jd_ut, _PLANET_BY_NAME[key])[4]
    return fixstar_mag(name)[0]


def _object_altitude(name, jd_ut, site):
    from . import eclipse_attrs
    key = name.strip().lower()
    tt = _s._tt_from_ut(jd_ut)
    if key in _PLANET_BY_NAME:
        return eclipse_attrs._horizon(_s._spk(), _s._PLANET_ID_CENTRES[_PLANET_BY_NAME[key]], tt, site, jd_ut)[2]
    st = _s.fixstar_ut(name, jd_ut, _c.FLG_SWIEPH | _c.FLG_EQUATORIAL)[0]
    ra, dec = np.radians(st[0]), np.radians(st[1])
    lst = (math.degrees(_s._gast_rad(jd_ut, tt)) + site[0]) % 360.0
    ha = math.radians(lst) - ra
    lat = math.radians(site[1])
    alt = math.degrees(math.asin(math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)))
    return alt + _refraction_true(alt, 1013.25, 15.0) if alt > -1.0 else alt


def _sun_altitude(jd_ut, site):
    from . import eclipse_attrs
    return eclipse_attrs._horizon(_s._spk(), _s._PLANET_ID_CENTRES[_c.SUN], _s._tt_from_ut(jd_ut), site, jd_ut)[1]


def _event_kind(event_type):
    """(is_morning, is_first_appearance) for the event types."""
    return {1: (True, True), 2: (False, False), 3: (False, True), 4: (True, False), 5: (False, True), 6: (True, False)}[int(event_type)]


def _visible_on(name, day_start, site, morning, arcus):
    """Is the object visible on the morning or evening beginning near `day_start` (UT)? Returns (visible, start, end) or None."""
    # scan the twilight for the moment the Sun is `arcus` degrees below the horizon
    step = 1.0 / 288.0
    t = day_start
    best = None
    prev = _sun_altitude(t, site)
    for _ in range(288 * 2):
        t2 = t + step
        cur = _sun_altitude(t2, site)
        down = prev > -arcus >= cur       # Sun sinking through the threshold (evening)
        up = prev < -arcus <= cur         # Sun rising through the threshold (morning)
        if (morning and up) or (not morning and down):
            lo, hi = t, t2
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                a = _sun_altitude(mid, site)
                if (a < -arcus) == (prev < -arcus):
                    lo = mid
                else:
                    hi = mid
            best = 0.5 * (lo + hi)
            break
        t, prev = t2, cur
    if best is None:
        return None
    return best


def heliacal_ut(tjdut, geopos, datm, dobs, objname, event_type, helflag=_c.FLG_SWIEPH, *args):
    """Next heliacal event of a planet or star. Returns (start of visibility, best moment, end of visibility), UT.
    Event types: 1 heliacal rising (morning first), 2 heliacal setting (evening last), 3 evening first, 4 morning last.
    Model: Ptolemy's arcus visionis (classical, documented in docs/SWISS_COMPATIBILITY.md); the atmosphere and observer
    arguments are accepted for compatibility but do not change the result."""
    site = _site(geopos)
    if int(event_type) not in (1, 2, 3, 4):
        raise Error("event type %s is not one of 1 (heliacal rising), 2 (heliacal setting), 3 (evening first), 4 (morning last)" % event_type)
    morning, first = _event_kind(event_type)
    key = objname.strip().lower()
    if key in ("sun", "moon"):
        raise Error("heliacal events are for planets and stars")
    if event_type in (3, 4) and key not in ("mercury", "venus"):
        raise Error("%s (event type %d) does not exist for %s" % ("evening first" if event_type == 3 else "morning last", event_type, key))
    mag = _object_magnitude(objname, float(tjdut))
    arcus = _arcus_visionis(objname, mag)
    jd0 = float(tjdut)
    prev_visible = None
    for day in range(0, 800):
        base = jd0 + day
        moment = _visible_on(objname, base - (0.5 if not morning else 0.0), site, morning, arcus)
        if moment is None:
            prev_visible = None          # a gap (polar summer): do not compare across it
            continue
        alt = _object_altitude(objname, moment, site)
        visible = alt >= 0.0
        if prev_visible is not None:
            appears = visible and not prev_visible          # becomes visible
            vanishes = (not visible) and prev_visible       # stops being visible
            wanted = appears if first else vanishes
            if wanted:
                t_event = moment
                # visibility window: from the object clearing the horizon to the Sun reaching the threshold depression
                step = 1.0 / 1440.0
                if morning:
                    t = moment - 0.1
                    while t < moment and _object_altitude(objname, t, site) < 0.0:
                        t += step
                    start, end = t, moment
                else:
                    t = moment + 0.1
                    while t > moment and _object_altitude(objname, t, site) < 0.0:
                        t -= step
                    start, end = moment, t
                if end < start:
                    start, end = end, start
                return (start, 0.5 * (start + end), end)
        prev_visible = visible
    raise Error("no %s found within three years" % ("heliacal rising" if event_type == 1 else "heliacal event"))


def heliacal_pheno_ut(tjdut, geopos, datm, dobs, objname, event_type, helflag=_c.FLG_SWIEPH):
    """Numbers behind a heliacal event at a moment: object altitude, Sun altitude, slots: 0 the depression of the Sun
    needed (arcus visionis), 1 object altitude, 2 object magnitude, 4 Sun altitude, 5 how much deeper the Sun is than needed.
    A 50-slot tuple, the rest zero."""
    site = _site(geopos)
    mag = _object_magnitude(objname, float(tjdut))
    arcus = _arcus_visionis(objname, mag)
    alt = _object_altitude(objname, float(tjdut), site)
    sun_alt = _sun_altitude(float(tjdut), site)
    return (arcus, alt, mag, 0.0, sun_alt, -sun_alt - arcus) + (0.0,) * 44


def vis_limit_mag(tjdut, geopos, datm, dobs, objname, helflag=_c.FLG_SWIEPH):
    """Faintest magnitude that can be seen at a moment, from the Sun's depression. Returns (1.0, 8 numbers):
    limiting magnitude, object altitude, object azimuth-free slots as zeros, Sun altitude, object magnitude.
    Classical model: an object of magnitude m needs the Sun at least `arcus visionis(m)` below the horizon (Ptolemy), which
    inverts to a limiting magnitude of 1 + (depression - 12) between about -4.6 (day) and 6.5 (dark sky)."""
    site = _site(geopos)
    sun_alt = _sun_altitude(float(tjdut), site)
    depression = max(-sun_alt, 0.0)
    limit = max(-4.6, min(6.5, 1.0 + (depression - 12.0)))
    obj_alt = _object_altitude(objname, float(tjdut), site)
    flag = 0 if sun_alt > -6.0 else (-2.0 if obj_alt < 0.0 else 1.0)
    return flag, (float(limit), float(obj_alt), 0.0, float(sun_alt), 0.0, 0.0, 0.0, float(_object_magnitude(objname, float(tjdut))), 0.0, 0.0)


def csroundsec(cs):
    """Round centiseconds to whole seconds, but never up onto a multiple of 30 degrees (29 59 59.5 stays 29 59 59)."""
    cs = int(cs)
    r = (cs + 50) // 100 * 100
    if r > cs and r % (30 * 360000) == 0 and cs % (30 * 360000) >= 30 * 360000 - 50:
        r -= 100
    return r


for _n in ("heliacal_ut", "heliacal_pheno_ut", "vis_limit_mag"):
    globals()[_n] = _wrap(globals()[_n])
__all__ += ["heliacal_ut", "heliacal_pheno_ut", "vis_limit_mag", "csroundsec", "HELIACAL_RISING", "HELIACAL_SETTING", "MORNING_FIRST",
            "EVENING_LAST", "EVENING_FIRST", "MORNING_LAST", "ACRONYCHAL_RISING", "ACRONYCHAL_SETTING", "COSMICAL_SETTING",
            "AUNIT_TO_KM", "AUNIT_TO_LIGHTYEAR", "AUNIT_TO_PARSEC", "PHOTOPIC_FLAG", "SCOTOPIC_FLAG", "MIXEDOPIC_FLAG", "TIDAL_AUTOMATIC",
            "TIDAL_DEFAULT", "TIDAL_DE200", "TIDAL_DE403", "TIDAL_DE404", "TIDAL_DE405", "TIDAL_DE406", "TIDAL_DE421", "TIDAL_DE430",
            "TIDAL_DE431", "TIDAL_DE441", "TIDAL_26"]

calc_pctr = _wrap(calc_pctr)
__all__.append("calc_pctr")
gauquelin_sector = _wrap(gauquelin_sector)
__all__ += ["gauquelin_sector"]
__all__ += ["get_orbital_elements", "orbit_max_min_true_distance", "nod_aps", "nod_aps_ut"]
