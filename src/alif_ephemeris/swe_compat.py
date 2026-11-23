"""A Swiss-style function interface on top of Alif Ephemeris.

Software written against the common `swisseph` module (functions such as `calc_ut`, `houses_ex`, `julday`
and constants such as `SUN` and `FLG_SPEED`) can use Alif Ephemeris instead by installing this module under
that name:

    import sys, alif_ephemeris.swe_compat as swe
    sys.modules["swisseph"] = swe

The module keeps a little global state (data file, observer site, sidereal mode, Delta-T override) because the
interface it imitates does. The library underneath stays stateless. Where this adapter cannot do something it
raises `Error` and says so; it never returns a made-up number.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

import numpy as np

from . import ayanamsa as _ay
from . import smallbodies as _sb
from . import hypothetical as _hyp
from . import frames, houses as _houses, lunar_points, planets, positions, sidereal_time, stars, timescales
from .ephemeris import Ephemeris
from .swe_constants import *          # noqa: F401,F403  (flag bits, body numbers, mode numbers)
from . import swe_constants as _c

__all__ = [n for n in dir(_c) if n.isupper()] + [
    "Error", "set_ephe_path", "close", "julday", "revjul", "deltat", "set_delta_t_userdef", "calc", "calc_ut",
    "houses", "houses_ex", "houses_armc", "sidtime", "sidtime0", "set_sid_mode", "get_ayanamsa",
    "get_ayanamsa_ut", "get_ayanamsa_ex", "get_ayanamsa_ex_ut", "get_ayanamsa_name", "fixstar", "fixstar_ut",
    "fixstar2", "fixstar2_ut", "cotrans", "cotrans_sp", "azalt", "azalt_rev", "set_topo", "degnorm", "split_deg",
    "get_planet_name", "get_library_path", "version", "house_name", "difdeg2n", "difdegn", "difcs2n", "difcsn",
    "day_of_week", "date_conversion", "lun_eclipse_when", "sol_eclipse_when_glob", "sol_eclipse_when_loc", "rise_trans", "pheno", "pheno_ut", "utc_to_jd", "jdet_to_utc", "jdut1_to_utc", "time_equ", "lat_to_lmt", "lmt_to_lat",
]

version = "alif-ephemeris compatibility interface"


class Error(Exception):
    """Raised for anything this interface cannot compute."""


# ---------------------------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------------------------
class _State:
    ephe_file = None
    topo = None                    # (east longitude, latitude, height m)
    delta_t_user = None            # days, or None for automatic
    sid_mode = 0
    sid_t0 = 0.0
    sid_ayan_t0 = 0.0


_S = _State()


_DATA_FILES_CACHE: dict = {}


def _data_files():
    """The planetary data files to use (remembered per setting, so the disk is not asked on every call)."""
    hit = _DATA_FILES_CACHE.get("files")
    if hit is None:
        hit = _DATA_FILES_CACHE["files"] = _find_data_files()
    return hit


def _find_data_files():
    """The planetary data files to use, most trusted first: the DE440 file for 1550 to 2650, then (if they are in the
    same folder) the two DE441 parts for the rest. Files named in ALIF_EPHEMERIS_FILE are used exactly as given."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    norm = lambda p: os.path.normcase(os.path.abspath(p))
    explicit = [p for p in os.environ.get("ALIF_EPHEMERIS_FILE", "").split(os.pathsep) if p and os.path.isfile(p)]
    if explicit:
        out, seen = [], set()
        for p in explicit:
            if norm(p) not in seen:
                seen.add(norm(p))
                out.append(p)
        return tuple(out)
    folders = []
    if _S.ephe_file:
        folders.append(os.path.dirname(_S.ephe_file))
    folders += [os.path.join(root, "data", "raw"), os.getcwd()]
    main = None
    if _S.ephe_file and os.path.isfile(_S.ephe_file):
        main = _S.ephe_file
    for folder in folders:
        if main:
            break
        for name in ("de440.bsp", "de440s.bsp"):
            f = os.path.join(folder, name)
            if os.path.isfile(f):
                main = f
                break
    if main is None:
        raise Error("no planetary data file found: set ALIF_EPHEMERIS_FILE to a JPL DE440 .bsp file")
    found, seen = [main], {norm(main)}
    for folder in [os.path.dirname(main)] + folders:            # long range files, when present
        for name in ("de441_part-1.bsp", "de441_part-2.bsp"):
            f = os.path.join(folder, name)
            if os.path.isfile(f) and norm(f) not in seen:
                try:
                    if os.path.getsize(f) > 1_000_000_000:       # skip partial downloads
                        seen.add(norm(f))
                        found.append(f)
                except OSError:
                    pass
    return tuple(found)


def _data_file():
    return _data_files()[0]


@lru_cache(maxsize=4)
def _eph(paths):
    return Ephemeris(list(paths))


def _spk():
    return _eph(_data_files()).spk


def set_ephe_path(path=None):
    """Remember a folder or file; a .bsp planetary file found there is used."""
    _DATA_FILES_CACHE.clear()
    _RESULT_CACHE.clear()
    _WINDOWS.clear()
    if path and os.path.isfile(path):
        _S.ephe_file = path
    elif path and os.path.isdir(path):
        for name in ("de440.bsp", "de441.bsp", "de440s.bsp"):
            f = os.path.join(path, name)
            if os.path.isfile(f):
                _S.ephe_file = f
                break


def close():
    _eph.cache_clear()
    _DATA_FILES_CACHE.clear()
    _RESULT_CACHE.clear()
    _WINDOWS.clear()


def set_topo(lon, lat, alt=0.0):
    _S.topo = (float(lon), float(lat), float(alt))


def set_delta_t_userdef(dt_days):
    """Override Delta-T (in days). Pass DELTAT_AUTOMATIC to go back to the built-in values."""
    _S.delta_t_user = None if dt_days <= _c.DELTAT_AUTOMATIC * 0.5 and dt_days < 0 else float(dt_days)


def deltat(jd_ut):
    """Delta-T in days."""
    if _S.delta_t_user is not None:
        return _S.delta_t_user
    return float(timescales.delta_t(jd_ut)) / 86400.0


# ---------------------------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------------------------
def julday(year, month, day, hour=0.0, gregflag=_c.GREG_CAL):
    return timescales.julian_day(int(year), int(month), float(day), float(hour),
                                 "gregorian" if gregflag == _c.GREG_CAL else "julian")


def revjul(jd, gregflag=_c.GREG_CAL):
    y, m, d, h = timescales.calendar_date(float(jd), "gregorian" if gregflag == _c.GREG_CAL else "julian")
    return y, m, d, h


def day_of_week(jd):
    return int(np.floor(jd + 0.5)) % 7


def date_conversion(year, month, day, utime, c="g"):
    jd = julday(year, month, day, utime, _c.GREG_CAL if c == "g" else _c.JUL_CAL)
    return True, jd


def degnorm(x):
    return x % 360.0


def difdegn(a, b):
    return (a - b) % 360.0


def difdeg2n(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def difcsn(a, b):
    return (a - b) % 360.0


def difcs2n(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def split_deg(ddeg, roundflag=0):
    sign = -1 if ddeg < 0 else 1
    x = abs(ddeg)
    deg = int(x)
    mn = int((x - deg) * 60)
    sec = ((x - deg) * 60 - mn) * 60
    isec = int(sec)
    return deg * sign if deg else 0, mn, isec, sec - isec, sign


# ---------------------------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------------------------
def _tt_from_ut(jd_ut):
    return jd_ut + deltat(jd_ut)


def utc_to_jd(y, mo, d, h, mi, s, gregflag=_c.GREG_CAL):
    jd_ut = julday(y, mo, d, h + mi / 60.0 + s / 3600.0, gregflag)
    tt = _tt_from_ut(jd_ut)
    return tt, jd_ut


def jdet_to_utc(jd_et, gregflag=_c.GREG_CAL):
    jd_ut = jd_et - deltat(jd_et)
    y, m, d, h = revjul(jd_ut, gregflag)
    hh = int(h)
    mm = int((h - hh) * 60)
    return y, m, d, hh, mm, ((h - hh) * 60 - mm) * 60


def jdut1_to_utc(jd_ut, gregflag=_c.GREG_CAL):
    y, m, d, h = revjul(jd_ut, gregflag)
    hh = int(h)
    mm = int((h - hh) * 60)
    return y, m, d, hh, mm, ((h - hh) * 60 - mm) * 60


_EE_WINDOWS: dict = {}
_EE_LEVELS = [(16.0, 56), (4.0, 24)]


def _ee_deps(tt):
    """(equation of the equinoxes, nutation in obliquity), radians, for one Terrestrial Time. Served from short
    Chebyshev windows that agree with the full series to 1e-5 arcsec; the full series is used when a window does not
    fit."""
    tt = float(tt)
    for li, (span, n) in enumerate(_EE_LEVELS):
        k = math.floor((tt - 2451545.0) / span)
        key = (li, k)
        win = _EE_WINDOWS.get(key)
        if win is None:
            if len(_EE_WINDOWS) > 20000:
                _EE_WINDOWS.clear()
            h = span / 2.0 + 0.01
            centre = 2451545.0 + (k + 0.5) * span
            xn = np.cos(np.pi * (np.arange(n) + 0.5) / n)
            xs = np.concatenate([xn, _CHECK])
            times = centre + xs * h
            ee = np.asarray(sidereal_time.equation_of_equinoxes(times), dtype=float)
            deps = np.asarray(frames.nutation_iau2000a(times)[1], dtype=float)
            coef = np.polynomial.chebyshev.chebfit(xn, np.stack([ee[:n], deps[:n]], axis=1), n - 1)
            fit = np.polynomial.chebyshev.chebval(_CHECK, coef)
            err = max(np.abs(fit[0] - ee[n:]).max(), np.abs(fit[1] - deps[n:]).max()) / 4.848136811e-6
            win = (centre, h, coef) if err < 1e-5 else False
            _EE_WINDOWS[key] = win
        if win is False:
            continue
        centre, h, coef = win
        x = (tt - centre) / h
        if -1.0 <= x <= 1.0:
            ks = _KS.get(coef.shape[0])
            if ks is None:
                ks = _KS[coef.shape[0]] = np.arange(coef.shape[0])
            out = np.cos(ks * math.acos(x)) @ coef
            return float(out[0]), float(out[1])
    dpsi_eps = frames.nutation_iau2000a(tt)
    return float(sidereal_time.equation_of_equinoxes(tt)), float(dpsi_eps[1])


def _gast_rad(jd_ut, tt):
    """Greenwich apparent sidereal time (radians) using the windowed equation of the equinoxes."""
    return (float(sidereal_time.gmst(jd_ut, tt)) + _ee_deps(tt)[0]) % (2.0 * math.pi)


def _true_obliquity_rad(tt):
    return float(frames.mean_obliquity(tt)) + _ee_deps(tt)[1]


def _st_offset_deg(jd_ut):
    """Degrees added to sidereal time. Always zero: sidereal time is the exact IAU value."""
    return 0.0


def _sidtime_exact(jd_ut):
    """Sidereal time in hours, exact IAU value, no reference-program convention."""
    tt = _tt_from_ut(jd_ut)
    return (math.degrees(_gast_rad(jd_ut, tt)) + _st_offset_deg(jd_ut)) / 15.0


def sidtime(jd_ut):
    """Greenwich apparent sidereal time in hours."""
    tt = _tt_from_ut(jd_ut)
    return (math.degrees(_gast_rad(jd_ut, tt)) + _st_offset_deg(jd_ut)) / 15.0


def sidtime0(jd_ut, eps, nut):
    tt = _tt_from_ut(jd_ut)
    return (float(np.degrees(sidereal_time.gmst(jd_ut, tt)) + nut * np.cos(np.radians(eps))) + _st_offset_deg(jd_ut)) / 15.0


def time_equ(jd_ut):
    """Equation of time in days (apparent minus mean solar time)."""
    tt = _tt_from_ut(jd_ut)
    w, f = timescales.split_jd(tt)
    sun = positions.observe(_spk(), planets.SUN, float(w), float(f), equatorial=True)
    ra = float(np.degrees(sun[0]))
    gmst = float(np.degrees(sidereal_time.gmst(jd_ut, tt)))
    # apparent solar time angle = GAST - RA; mean solar time angle = UT*15 + 180
    ha = (math.degrees(_gast_rad(jd_ut, tt)) - ra - 180.0) % 360.0
    mean = ((jd_ut + 0.5) % 1.0) * 360.0
    return (((ha - mean) + 180.0) % 360.0 - 180.0) / 360.0


def lmt_to_lat(jd_lmt, geolon):
    jd_ut = jd_lmt - geolon / 360.0
    return True, jd_lmt + time_equ(jd_ut)


def lat_to_lmt(jd_lat, geolon):
    jd_lmt = jd_lat
    for _ in range(4):
        jd_lmt = jd_lat - time_equ(jd_lmt - geolon / 360.0)
    return True, jd_lmt


# ---------------------------------------------------------------------------------------------
# sidereal modes
# ---------------------------------------------------------------------------------------------
_SIDM_NAMES = {
    0: "Fagan-Bradley", 1: "Lahiri", 2: "DeLuce", 3: "Raman", 4: "Usha-Shashi", 5: "Krishnamurti (KP)",
    6: "Djwhal Khul", 7: "Yukteshwar", 8: "JN Bhasin", 9: "Babylonian (Kugler 1)", 10: "Babylonian (Kugler 2)",
    11: "Babylonian (Kugler 3)", 12: "Babylonian (Huber)", 13: "Babylonian (Eta Piscium)",
    14: "Aldebaran 15 Taurus (fixed)", 15: "Hipparchos", 16: "Sassanian", 17: "Galactic Centre 0 Sagittarius",
    18: "J2000", 19: "J1900", 20: "B1950", 21: "Suryasiddhanta", 22: "Suryasiddhanta (mean Sun)", 23: "Aryabhata",
    24: "Aryabhata (mean Sun)", 25: "Suryasiddhanta (Revati)", 26: "Suryasiddhanta (Citra)",
    27: "True Chitra (Spica 180)", 28: "True Revati (Zeta Piscium 359-50)", 29: "True Pushya (Delta Cancri 106)",
    34: "Galactic (Mardyks)", 35: "True Mula (Lambda Scorpii 240)", 37: "Aryabhata 522", 38: "Babylonian (Britton)",
    40: "Galactic Centre 0 Capricorn (Cochrane)", 41: "Galactic equator (Fiorenza)", 42: "Valens (Moon)",
    43: "Lahiri 1940", 44: "Lahiri (VP285)", 45: "Krishnamurti (VP291)", 46: "Lahiri (ICRC)",
}


def set_sid_mode(mode, t0=0.0, ayan_t0=0.0):
    _S.sid_mode = int(mode) & 0xFF
    _S.sid_t0 = float(t0)
    _S.sid_ayan_t0 = float(ayan_t0)


def get_ayanamsa_name(mode):
    return _SIDM_NAMES.get(int(mode), "unknown")


def _mean_ayanamsa(tt):
    mode = _S.sid_mode
    if mode == _c.SIDM_USER:
        return float(_ay.fixed_point_ayanamsa(_S.sid_ayan_t0, _S.sid_t0 or 2451545.0, tt))
    name = _SIDM_NAMES.get(mode)
    if name is None:
        raise Error(f"sidereal mode {mode} is not available in Alif Ephemeris")
    spec = _ay.get(name)
    if spec.kind == "star" and name != "Lahiri (JHora)" and _CONVENTION["outer_barycentres"]:
        return _star_ayanamsa_with_aberration(spec, tt)
    return float(_ay.ayanamsa(name, tt))


def _star_ayanamsa_with_aberration(spec, tt):
    """The reference program holds a star (or the galactic centre) at its target using the apparent position with
    annual aberration but without nutation. The core library's value leaves the aberration out."""
    tt = float(tt)
    if isinstance(spec.star, tuple):
        ra, dec = np.radians(spec.star[0]), np.radians(spec.star[1])
        d = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    else:
        lon0, lat0 = stars.mean_ecliptic(stars.find(spec.star), tt)
        lo, la = np.radians(float(lon0)), np.radians(float(lat0))
        ecl = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        eps = float(frames.mean_obliquity(tt))
        d = np.array([ecl[0], np.cos(eps) * ecl[1] - np.sin(eps) * ecl[2], np.sin(eps) * ecl[1] + np.cos(eps) * ecl[2]])
        gamb, phib, psib, epsa = frames.fukushima_williams(tt)
        pm = frames._rot_x(-epsa) @ frames._rot_z(-psib) @ frames._rot_x(phib) @ frames._rot_z(gamb)
        d = pm.T @ d                                               # mean equator of date back to ICRS directions
    gamb, phib, psib, epsa = frames.fukushima_williams(tt)
    pm = frames._rot_x(-epsa) @ frames._rot_z(-psib) @ frames._rot_x(phib) @ frames._rot_z(gamb)
    whole, frac = (float(x) for x in timescales.split_jd(tt))
    _, v = planets._barycentric(_spk(), planets.EARTH, whole, frac)
    beta = np.asarray(v, dtype=float) / 299792.458                # Earth's velocity over c, ICRS
    d_app = d + beta - float(d @ beta) * d                         # first-order annual aberration
    eq = pm @ (d_app / np.linalg.norm(d_app))
    lon, _, _ = frames.to_spherical(frames.equator_to_ecliptic(eq, frames.mean_obliquity(tt)))
    return float((np.degrees(lon) - spec.target + 180.0) % 360.0 - 180.0)


def get_ayanamsa_ex(jd_et, flags=0):
    ay = _mean_ayanamsa(jd_et)
    if not (flags & _c.FLG_NONUT):
        ay += float(np.degrees(frames.nutation_iau2000a(jd_et)[0]))
    return flags, ay % 360.0


def get_ayanamsa_ex_ut(jd_ut, flags=0):
    return get_ayanamsa_ex(_tt_from_ut(jd_ut), flags)


def get_ayanamsa(jd_et):
    return get_ayanamsa_ex(jd_et, _c.FLG_NONUT)[1]


def get_ayanamsa_ut(jd_ut):
    return get_ayanamsa_ex_ut(jd_ut, _c.FLG_NONUT)[1]


# ---------------------------------------------------------------------------------------------
# planets
# ---------------------------------------------------------------------------------------------
_PLANET_ID = {_c.SUN: planets.SUN, _c.MOON: planets.MOON, _c.MERCURY: planets.MERCURY, _c.VENUS: planets.VENUS,
              _c.MARS: planets.MARS, _c.JUPITER: planets.JUPITER, _c.SATURN: planets.SATURN,
              _c.URANUS: planets.URANUS, _c.NEPTUNE: planets.NEPTUNE, _c.PLUTO: planets.PLUTO,
              _c.EARTH: planets.EARTH}
_PLANET_ID_CENTRES = dict(_PLANET_ID)

_CONVENTION = {"outer_barycentres": True}
_CENTRE_IDS = {}


def set_reference_convention(on=True):
    """Follow the reference program's conventions for the outer planets (default on): Jupiter to Pluto are the
    barycentres of their systems, and Uranus and Neptune come from the plain DE440 barycentres. Off gives the
    physically better planet centres with NASA's newest Uranus and Neptune orbits (what NASA Horizons shows)."""
    _CONVENTION["outer_barycentres"] = bool(on)
    if on:
        _PLANET_ID.update({_c.JUPITER: 5, _c.SATURN: 6, _c.URANUS: 7007, _c.NEPTUNE: 7008, _c.PLUTO: 9})
    else:
        _PLANET_ID.update({_c.JUPITER: planets.JUPITER, _c.SATURN: planets.SATURN, _c.URANUS: planets.URANUS,
                           _c.NEPTUNE: planets.NEPTUNE, _c.PLUTO: planets.PLUTO})
    _RESULT_CACHE.clear()
    _WINDOWS.clear()


_SMALL_BODY = {_c.CHIRON: _sb.CHIRON, _c.PHOLUS: _sb.PHOLUS, _c.CERES: _sb.CERES, _c.PALLAS: _sb.PALLAS,
               _c.JUNO: _sb.JUNO, _c.VESTA: _sb.VESTA}

_NAMES = {0: "Sun", 1: "Moon", 2: "Mercury", 3: "Venus", 4: "Mars", 5: "Jupiter", 6: "Saturn", 7: "Uranus",
          8: "Neptune", 9: "Pluto", 10: "mean Node", 11: "true Node", 12: "mean Apogee", 13: "osc. Apogee",
          14: "Earth", 15: "Chiron", 16: "Pholus", 17: "Ceres", 18: "Pallas", 19: "Juno", 20: "Vesta",
          21: "intp. Apogee", 22: "intp. Perigee"}


def get_planet_name(ipl):
    return _NAMES.get(ipl, "body %d" % ipl)


def _observer_kwargs(flags):
    kw = {}
    kw["light_time"] = not (flags & _c.FLG_TRUEPOS)
    kw["deflection"] = not (flags & (_c.FLG_TRUEPOS | _c.FLG_NOGDEFL))
    kw["aberration"] = not (flags & (_c.FLG_TRUEPOS | _c.FLG_NOABERR))
    kw["nutation"] = not (flags & _c.FLG_NONUT)
    kw["frame"] = "j2000" if flags & _c.FLG_J2000 else "date"
    kw["equatorial"] = bool(flags & (_c.FLG_EQUATORIAL))
    if flags & _c.FLG_HELCTR:
        kw["observer"] = "sun"
    elif flags & _c.FLG_BARYCTR:
        kw["observer"] = "ssb"
    elif flags & _c.FLG_TOPOCTR:
        if _S.topo is None:
            raise Error("topocentric positions need set_topo() first")
        kw["observer"] = "site"
        kw["site"] = _S.topo
    return kw


def _sidereal_ayanamsa(tt, flags):
    mean = _mean_ayanamsa(tt)
    if flags & _c.FLG_NONUT:
        return mean
    return mean + float(np.degrees(frames.nutation_iau2000a(tt)[0]))


def _planet_lonlat_dist(body, tt_whole, tt_frac, flags, jd_ut):
    """(lon, lat, dist) in degrees/degrees/AU for arrays of times, honouring the flags."""
    kw = _observer_kwargs(flags)
    spk = _spk()
    if body in _PLANET_ID:
        pid = _PLANET_ID[body]
        if pid == planets.EARTH and "observer" not in kw:
            z = np.zeros_like(np.asarray(tt_frac, dtype=float))
            return z, z, z
        if pid == planets.EARTH:
            pass
        if kw.get("observer") == "site":
            kw["jd_ut"] = jd_ut
        lon, lat, dist = positions.observe(spk, pid, tt_whole, tt_frac, **kw)
        lon_d, lat_d = np.degrees(lon), np.degrees(lat)
        return lon_d, lat_d, dist
    if body in (_c.MEAN_NODE, _c.MEAN_APOG, _c.TRUE_NODE, _c.OSCU_APOG, _c.INTP_APOG, _c.INTP_PERG):
        lon, lat, dist = _lunar_point(body, tt_whole, tt_frac, flags, spk)
        if flags & _c.FLG_EQUATORIAL:
            tt = np.asarray(tt_whole, dtype=float) + np.asarray(tt_frac, dtype=float)
            eps = np.degrees(frames.mean_obliquity(tt) if flags & _c.FLG_NONUT else frames.true_obliquity(tt))
            lo, la = np.radians(lon), np.radians(lat)
            x, y, z = np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)
            e = np.radians(eps)
            lon = np.degrees(np.arctan2(y * np.cos(e) - z * np.sin(e), x)) % 360.0
            lat = np.degrees(np.arcsin(np.clip(y * np.sin(e) + z * np.cos(e), -1, 1)))
        return lon, lat, dist
    if False:
        tt = np.asarray(tt_whole, dtype=float) + np.asarray(tt_frac, dtype=float)
        dpsi = np.degrees(frames.nutation_iau2000a(tt)[0])
        if body == _c.MEAN_NODE:
            lon, dist = _compat_node(tt) + dpsi, 0.0025695
        elif body == _c.MEAN_APOG:
            mlon, mlat = lunar_points.mean_apogee_lonlat(tt)
            lon, dist = mlon + dpsi, 0.0027097
            if flags & _c.FLG_NONUT:
                lon = lon - dpsi
            return lon, mlat, dist * np.ones_like(lon)
        elif body == _c.TRUE_NODE:
            lon, dist = lunar_points.true_node(spk, tt_whole, tt_frac), 0.0025695
        else:
            lon, lat = lunar_points.osculating_apogee(spk, tt_whole, tt_frac)
            return lon, lat, 0.0027097 * np.ones_like(lon)
        if flags & _c.FLG_NONUT:
            lon = lon - (dpsi if body in (_c.MEAN_NODE, _c.MEAN_APOG) else 0.0)
        return lon, np.zeros_like(lon), dist * np.ones_like(lon)
    if _hyp.supports(body + _hyp.OFFSET):
        if kw.get("observer") == "site":
            kw["jd_ut"] = jd_ut
        lon, lat, dist = positions.observe(spk, body + _hyp.OFFSET, tt_whole, tt_frac, **kw)
        return np.degrees(lon), np.degrees(lat), dist
    sb = _SMALL_BODY.get(body, body + _sb.OFFSET - _c.AST_OFFSET if body > _c.AST_OFFSET else None)
    if sb is not None and _sb.supports(sb):
        if kw.get("observer") == "site":
            kw["jd_ut"] = jd_ut
        lon, lat, dist = positions.observe(spk, sb, tt_whole, tt_frac, **kw)
        return np.degrees(lon), np.degrees(lat), dist
    raise Error(f"body {body} is not available in Alif Ephemeris yet")


def _lunar_point(body, tt_whole, tt_frac, flags, spk):
    tt = np.asarray(tt_whole, dtype=float) + np.asarray(tt_frac, dtype=float)
    if flags & (_c.FLG_HELCTR | _c.FLG_BARYCTR):                # the reference gives zeros for these points seen from the Sun or the barycentre
        z = np.zeros_like(tt)
        return z, z.copy(), z.copy()
    dpsi = np.degrees(frames.nutation_iau2000a(tt)[0])
    nonut = bool(flags & _c.FLG_NONUT)
    if body == _c.MEAN_NODE:
        lon, lat, dist = _compat_node(tt) + (0.0 if nonut else dpsi), np.zeros_like(tt), 0.0025695
    elif body == _c.MEAN_APOG:
        mlon, lat = lunar_points.mean_apogee_lonlat(tt)
        lon, dist = mlon + (0.0 if nonut else dpsi), 0.0027097
    elif body == _c.TRUE_NODE:
        lon, lat = lunar_points.true_node(spk, tt_whole, tt_frac) - (dpsi if nonut else 0.0), np.zeros_like(tt)
        dist = lunar_points.true_node_distance(spk, tt_whole, tt_frac) / planets.AU_KM
    elif body in (_c.INTP_APOG, _c.INTP_PERG):
        fn = lunar_points.interpolated_perigee if body == _c.INTP_PERG else lunar_points.interpolated_apogee
        if np.ndim(tt) > 0 and np.size(tt) > 1:
            res = np.array([fn(spk, t) for t in np.ravel(tt)])
            lon, lat, dist = res[:, 0].reshape(np.shape(tt)), res[:, 1].reshape(np.shape(tt)), res[:, 2].reshape(np.shape(tt))
        else:
            lon, lat, dist = fn(spk, float(np.ravel(tt)[0]))
        dpsi_ = 0.0 if nonut else dpsi
        lon_o, lat_o = (np.asarray(lon) + dpsi_) % 360.0, np.asarray(lat) * np.ones_like(tt)
        if flags & _c.FLG_J2000:
            lon_o, lat_o = _ecliptic_date_to_j2000(lon_o - dpsi_, lat_o, tt)
        return lon_o % 360.0, lat_o, np.asarray(dist) * np.ones_like(tt)
    else:
        lon, lat = lunar_points.osculating_apogee(spk, tt_whole, tt_frac)
        lon = lon - (dpsi if nonut else 0.0)
        dist = 0.0027097
    lon, lat = np.asarray(lon, dtype=float) % 360.0, np.asarray(lat, dtype=float)
    if flags & _c.FLG_J2000:                                  # J2000 means no precession and no nutation
        lon, lat = _ecliptic_date_to_j2000(lon - (0.0 if nonut else dpsi), lat, tt)
    return lon % 360.0, lat, dist * np.ones_like(tt)


def _ecliptic_date_to_j2000(lon, lat, tt):
    """Longitude and latitude (degrees) on the mean ecliptic of date to the mean ecliptic of J2000 (IAU 2006)."""
    lon, lat, tt = np.broadcast_arrays(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float), np.asarray(tt, dtype=float))
    lo, la = np.radians(lon), np.radians(lat)
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)
    gamb, phib, psib, epsa = frames.fukushima_williams(tt)
    eps0 = 84381.406 * frames.ASEC2RAD
    ce, se = np.cos(epsa), np.sin(epsa)
    equ = np.stack([v[..., 0], ce * v[..., 1] - se * v[..., 2], se * v[..., 1] + ce * v[..., 2]], axis=-1)   # mean equator of date
    prec = frames._rot_x(-epsa) @ frames._rot_z(-psib) @ frames._rot_x(phib) @ frames._rot_z(gamb)
    gcrs = np.einsum("...ji,...j->...i", prec, equ)                                                          # transpose = inverse
    c0, s0 = np.cos(eps0), np.sin(eps0)
    ecl = np.stack([gcrs[..., 0], c0 * gcrs[..., 1] + s0 * gcrs[..., 2], -s0 * gcrs[..., 1] + c0 * gcrs[..., 2]], axis=-1)
    return np.degrees(np.arctan2(ecl[..., 1], ecl[..., 0])) % 360.0, np.degrees(np.arcsin(np.clip(ecl[..., 2], -1.0, 1.0)))


_RESULT_CACHE: dict = {}
_RESULT_CACHE_MAX = 50000


_SIMPLE_EXCLUDE = _c.FLG_SPEED | _c.FLG_SPEED3 | _c.FLG_SIDEREAL | _c.FLG_XYZ | _c.FLG_RADIANS | _c.FLG_TOPOCTR


def _calc_tt(jd_tt, body, flags, jd_ut=None):
    """Cached front door of the position calculation. A position depends on the time, body, flags and the settings
    held in `_S`, so all of them form the key; software that asks for the same place twice gets it back at once."""
    key = (float(jd_tt), int(body), int(flags), None if jd_ut is None else float(jd_ut), _S.topo, _S.delta_t_user,
           _S.sid_mode, _S.sid_t0, _S.sid_ayan_t0)
    hit = _RESULT_CACHE.get(key)
    if hit is None:
        hit = _simple_position(float(jd_tt), body, int(flags)) if not flags & _SIMPLE_EXCLUDE else None
        if hit is None:
            hit = _calc_tt_uncached(jd_tt, body, flags, jd_ut)
        if len(_RESULT_CACHE) >= _RESULT_CACHE_MAX:
            _RESULT_CACHE.clear()
        _RESULT_CACHE[key] = hit
    return hit


def _simple_position(jd_tt, body, flags):
    """The common case (one time, no speed, tropical, ecliptic): straight from a validated window, or None."""
    levels = _WINDOW_LEVELS.get(body)
    if levels is None:
        return None
    files = _data_files()
    for li, (span, n) in enumerate(levels):
        k = math.floor((jd_tt - 2451545.0) / span)
        key = (body, flags, files, li, k)
        win = _WINDOWS.get(key)
        if win is None:
            if len(_WINDOWS) >= _WINDOW_MAX:
                _WINDOWS.clear()
            win = _WINDOWS[key] = _build_window(body, flags, span, n, k)
        if win is False:
            continue
        c_whole, c_frac, h, coef = win
        x = ((jd_tt - c_whole) - c_frac) / h
        if x > 1.0 or x < -1.0:
            continue
        ks = _KS.get(coef.shape[0])
        if ks is None:
            ks = _KS[coef.shape[0]] = np.arange(coef.shape[0])
        out = np.cos(ks * math.acos(x)) @ coef
        return ((float(out[0]) % 360.0, float(out[1]), float(out[2]), 0.0, 0.0, 0.0), flags)
    return None


_WINDOWS: dict = {}
_WINDOW_MAX = 8000
# per body, window sizes to try from longest to shortest: (length in days, Chebyshev nodes). A window is kept only if
# it reproduces extra points computed the full way to 0.001 arcsec; otherwise the next shorter one is tried.
_WINDOW_LEVELS = {_c.MOON: [(16.0, 160), (2.0, 30)], _c.SUN: [(64.0, 64), (16.0, 24)], _c.MERCURY: [(16.0, 64), (4.0, 16)],
                  _c.VENUS: [(64.0, 96), (16.0, 48), (4.0, 16)], _c.MARS: [(32.0, 96), (8.0, 24)],
                  _c.JUPITER: [(64.0, 240), (8.0, 60)], _c.SATURN: [(32.0, 40), (8.0, 16)],
                  _c.URANUS: [(128.0, 64), (32.0, 32), (8.0, 16)], _c.NEPTUNE: [(128.0, 64), (32.0, 32), (8.0, 16)],
                  _c.PLUTO: [(128.0, 64), (32.0, 32), (8.0, 16)],
                  _c.MEAN_NODE: [(128.0, 24), (16.0, 16)], _c.MEAN_APOG: [(32.0, 48), (8.0, 24)],
                  _c.TRUE_NODE: [(16.0, 80), (4.0, 24)]}
_WINDOW_MARGIN = 0.03
_KS: dict = {}
_CHECK = np.array([-0.93, -0.41, 0.07, 0.52, 0.95])           # extra points, used to validate a fit


def _build_window(body, base, span, n, k):
    h = span / 2.0 + _WINDOW_MARGIN
    centre = 2451545.0 + (k + 0.5) * span
    c_whole = math.floor(centre)
    c_frac = centre - c_whole
    xn = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    nodes = c_frac + np.concatenate([xn, _CHECK]) * h
    lon, lat, dist = _planet_lonlat_dist(body, c_whole, nodes, base, c_whole + nodes)
    lon = np.degrees(np.unwrap(np.radians(np.asarray(lon, dtype=float))))
    lat, dist = np.asarray(lat, dtype=float), np.asarray(dist, dtype=float)
    cheb, ev0 = np.polynomial.chebyshev.chebfit, np.polynomial.chebyshev.chebval
    coef = cheb(xn, np.stack([lon[:n], lat[:n], dist[:n]], axis=1), n - 1)           # one (n, 3) block of coefficients
    fit = ev0(_CHECK, coef)                                                            # (3, 5)
    err = max(np.abs(fit[0] - lon[n:]).max(), np.abs(fit[1] - lat[n:]).max()) * 3600.0
    return (c_whole, c_frac, h, coef) if err < 1e-3 else False


def _fast_lld(body, tt_whole, tt_frac, flags, jd_ut):
    """Same answer as _planet_lonlat_dist, much faster when many nearby times are asked for in turn. A Chebyshev fit
    of the body's own computed position is built once per window of days and reused; each fit is checked against
    extra points computed the full way and used only if it agrees to 0.001 arcsec (tests/test_extras.py)."""
    levels = _WINDOW_LEVELS.get(body)
    if levels is None or flags & (_c.FLG_TOPOCTR | _c.FLG_XYZ | _c.FLG_RADIANS):
        return _planet_lonlat_dist(body, tt_whole, tt_frac, flags, jd_ut)
    fr = np.asarray(tt_frac, dtype=float)
    t = float(tt_whole) + float(np.min(fr))
    base = flags & ~(_c.FLG_SPEED | _c.FLG_SPEED3)
    files = _data_files()
    for li, (span, n) in enumerate(levels):
        k = math.floor((t - 2451545.0) / span)
        key = (body, base, files, li, k)
        win = _WINDOWS.get(key)
        if win is None:
            if len(_WINDOWS) >= _WINDOW_MAX:
                _WINDOWS.clear()
            win = _WINDOWS[key] = _build_window(body, base, span, n, k)
        if win is False:
            continue
        c_whole, c_frac, h, coef = win
        x = ((float(tt_whole) - c_whole) + fr - c_frac) / h
        if np.any(np.abs(x) > 1.0):
            continue
        ks = _KS.get(n_coef := coef.shape[0])
        if ks is None:
            ks = _KS[n_coef] = np.arange(n_coef)
        xa = x.reshape(-1) if hasattr(x, 'reshape') else np.array([x])
        out = np.cos(np.multiply.outer(np.arccos(np.clip(xa, -1.0, 1.0)), ks)) @ coef          # (points, 3)
        lon, lat, dist = out[:, 0] % 360.0, out[:, 1], out[:, 2]
        if np.ndim(x) == 0:
            return lon[0], lat[0], dist[0]
        return lon, lat, dist
    return _planet_lonlat_dist(body, tt_whole, tt_frac, flags, jd_ut)


def _calc_tt_uncached(jd_tt, body, flags, jd_ut=None):
    flags = int(flags)
    if body == _c.ECL_NUT:
        tt = float(jd_tt)
        dpsi, deps = frames.nutation_iau2000a(tt)
        mean = float(np.degrees(frames.mean_obliquity(tt)))
        return (mean + float(np.degrees(deps)), mean, float(np.degrees(dpsi)), float(np.degrees(deps)), 0.0, 0.0), flags
    whole, frac = timescales.split_jd(float(jd_tt))
    whole, frac = float(whole), float(frac)
    want_speed = bool(flags & (_c.FLG_SPEED | _c.FLG_SPEED3))
    if jd_ut is None:
        jd_ut = jd_tt - (deltat(jd_tt) if _S.delta_t_user is not None else float(timescales.delta_t(jd_tt)) / 86400.0)
    if want_speed:
        steps = np.array([0.0, 0.005, -0.005, 0.01, -0.01])
        fr = frac + steps
        uts = jd_ut + steps
    else:
        steps = np.array([0.0])
        fr = frac                      # one plain number: takes the fast single-time path
        uts = jd_ut
    lon, lat, dist = _fast_lld(body, whole, fr, flags, uts)
    lon, lat, dist = (np.asarray(a, dtype=float).reshape(-1) for a in (lon, lat, dist))
    if flags & _c.FLG_SIDEREAL:
        ay = np.array([_sidereal_ayanamsa(whole + f, flags) for f in np.atleast_1d(fr)])
        lon = lon - ay
    lon = lon % 360.0 if not (flags & _c.FLG_EQUATORIAL) else lon % 360.0
    out = [float(lon[0]), float(lat[0]), float(dist[0]), 0.0, 0.0, 0.0]
    if want_speed:
        def diff(i, j, h):
            dl = (lon[i] - lon[j] + 180.0) % 360.0 - 180.0
            return np.array([dl, lat[i] - lat[j], dist[i] - dist[j]]) / (2 * h)
        d = (4.0 * diff(1, 2, 0.005) - diff(3, 4, 0.01)) / 3.0
        out[3], out[4], out[5] = float(d[0]), float(d[1]), float(d[2])
    if flags & _c.FLG_XYZ:
        lo, la = np.radians(out[0]), np.radians(out[1])
        r = out[2]
        x = (r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la))
        out = [x[0], x[1], x[2], 0.0, 0.0, 0.0]
    elif flags & _c.FLG_RADIANS:
        out = [np.radians(out[0]), np.radians(out[1]), out[2], np.radians(out[3]), np.radians(out[4]), out[5]]
    return tuple(out), flags


def _compat_node(jd_tt):
    """Mean node: the standard series (IERS Conventions Delaunay argument for the Moon's node, mean equinox of date)."""
    return lunar_points.mean_node(jd_tt)


def calc(jd_et, ipl, iflag=_c.FLG_SWIEPH | _c.FLG_SPEED):
    """Position of a body at Ephemeris (Terrestrial) Time. Returns ((lon, lat, dist, dlon, dlat, ddist), flags)."""
    return _calc_tt(jd_et, ipl, iflag)


def calc_ut(jd_ut, ipl, iflag=_c.FLG_SWIEPH | _c.FLG_SPEED):
    """Position of a body at Universal Time. Returns ((lon, lat, dist, dlon, dlat, ddist), flags)."""
    return _calc_tt(_tt_from_ut(jd_ut), ipl, iflag, jd_ut=float(jd_ut))


# ---------------------------------------------------------------------------------------------
# houses
# ---------------------------------------------------------------------------------------------
def _hsys_name(hsys):
    if isinstance(hsys, (bytes, bytearray)):
        hsys = hsys.decode("ascii")
    if isinstance(hsys, int):
        hsys = chr(hsys)
    name = _houses.LETTERS.get(hsys)
    if name is None:
        raise Error(f"house system {hsys!r} is not available in Alif Ephemeris")
    return name


def _refuse_beyond_polar_circle(name, lat, eps):
    """The reference program refuses Placidus and Koch inside the polar circles (latitude 90 degrees minus the
    obliquity), because their definition fails there. We can compute something, but software written for the
    reference expects the error and falls back to another system, so the Swiss-style layer raises it too."""
    if name in ("Placidus", "Koch") and abs(lat) >= 90.0 - eps:
        raise Error(f"{name} houses cannot be computed at latitude {lat:.2f} (inside the polar circle)")


def _gauquelin_houses(jd_ut, lat, lon, tt, flags):
    ramc = (math.degrees(_gast_rad(jd_ut, tt)) + float(lon)) % 360.0 + _st_offset_deg(jd_ut)
    eps = math.degrees(_true_obliquity_rad(tt))
    try:
        cusps = _houses.gauquelin_sectors(ramc, eps, lat)
    except ValueError as exc:
        raise Error(str(exc))
    h = _houses.compute(ramc, eps, lat, "Porphyry")
    asc, mc, vtx = h.ascendant, h.midheaven, h.vertex
    ascmc = [asc, mc, ramc, vtx, float(_houses.ascendant_from(ramc, eps, 0.0)), 0.0, 0.0, 0.0]
    if flags & _c.FLG_SIDEREAL:
        ay = _sidereal_ayanamsa(tt, flags)
        cusps = [(c - ay) % 360.0 for c in cusps]
        for i in (0, 1, 3, 4):
            ascmc[i] = (ascmc[i] - ay) % 360.0
    return tuple(cusps), tuple(ascmc)


def houses_ex(jd_ut, lat, lon, hsys=b"P", flags=0):
    """House cusps and angles. Returns (cusps, ascmc): 12 cusp longitudes and 8 angle values (degrees)."""
    name = _hsys_name(hsys)
    tt = _tt_from_ut(jd_ut)
    if name == "Gauquelin":
        return _gauquelin_houses(jd_ut, lat, lon, tt, flags)
    _refuse_beyond_polar_circle(name, float(lat), math.degrees(_true_obliquity_rad(tt)))
    ramc = (math.degrees(_gast_rad(jd_ut, tt)) + float(lon)) % 360.0 + _st_offset_deg(jd_ut)
    eps = math.degrees(_true_obliquity_rad(tt))
    if flags & _c.FLG_NONUT:
        gm = float(np.degrees(sidereal_time.gmst(jd_ut, tt))) + _st_offset_deg(jd_ut)
        ramc = (gm + lon) % 360.0
        eps = float(np.degrees(frames.mean_obliquity(tt)))
    h = _houses.compute(ramc, eps, lat, name)
    cusps = list(h.cusps)
    asc, mc, vtx = h.ascendant, h.midheaven, h.vertex
    equasc = float(_houses.ascendant_from(ramc, eps, 0.0))
    ascmc = [asc, mc, ramc, vtx, equasc, 0.0, 0.0, 0.0]
    if flags & _c.FLG_SIDEREAL:
        ay = _sidereal_ayanamsa(tt, flags)
        cusps = [(c - ay) % 360.0 for c in cusps]
        ascmc[0] = (asc - ay) % 360.0
        ascmc[1] = (mc - ay) % 360.0
        ascmc[3] = (vtx - ay) % 360.0
        ascmc[4] = (equasc - ay) % 360.0
    return tuple(cusps), tuple(ascmc)


def houses(jd_ut, lat, lon, hsys=b"P"):
    return houses_ex(jd_ut, lat, lon, hsys, 0)


def houses_armc(armc, lat, eps, hsys=b"P"):
    name = _hsys_name(hsys)
    if name == "Gauquelin":
        try:
            cusps = _houses.gauquelin_sectors(armc, eps, lat)
        except ValueError as exc:
            raise Error(str(exc))
        h = _houses.compute(armc, eps, lat, "Porphyry")
        return tuple(cusps), (h.ascendant, h.midheaven, armc % 360.0, h.vertex, 0.0, 0.0, 0.0, 0.0)
    _refuse_beyond_polar_circle(name, float(lat), float(eps))
    h = _houses.compute(armc, eps, lat, name)
    return tuple(h.cusps), (h.ascendant, h.midheaven, armc % 360.0, h.vertex, 0.0, 0.0, 0.0, 0.0)


def house_name(hsys):
    return _hsys_name(hsys)


# ---------------------------------------------------------------------------------------------
# fixed stars
# ---------------------------------------------------------------------------------------------
def _star_key(name):
    key = str(name).strip()
    key = key.split(",")[0] if "," in key and key.split(",")[0].strip() else key.lstrip(",")
    return key.strip()


_STAR_WINDOWS: dict = {}
_STAR_LEVELS = [(32.0, 48), (8.0, 20)]


def _star_window(index, base, whole, frac, direct):
    """Apparent place of a star from a validated Chebyshev window (stars move smoothly), or the full calculation."""
    t = whole + frac
    ev = np.polynomial.chebyshev.chebval
    for li, (span, n) in enumerate(_STAR_LEVELS):
        k = math.floor((t - 2451545.0) / span)
        key = (index, base, _data_files(), li, k)
        win = _STAR_WINDOWS.get(key)
        if win is None:
            if len(_STAR_WINDOWS) > 20000:
                _STAR_WINDOWS.clear()
            h = span / 2.0 + _WINDOW_MARGIN
            centre = 2451545.0 + (k + 0.5) * span
            cw = math.floor(centre)
            cf = centre - cw
            xn = np.cos(np.pi * (np.arange(n) + 0.5) / n)
            xs = np.concatenate([xn, _CHECK])
            try:
                lo, la = direct(cw, cf + xs * h)
                lo = np.degrees(np.unwrap(np.radians(np.asarray(lo, dtype=float).reshape(-1))))
                la = np.asarray(la, dtype=float).reshape(-1)
                if lo.size != xs.size:
                    raise ValueError
                cl, cb = np.polynomial.chebyshev.chebfit(xn, lo[:n], n - 1), np.polynomial.chebyshev.chebfit(xn, la[:n], n - 1)
                err = max(np.abs(ev(_CHECK, cl) - lo[n:]).max(), np.abs(ev(_CHECK, cb) - la[n:]).max()) * 3600.0
                win = (cw, cf, h, cl, cb) if err < 1e-3 else False
            except Exception:
                win = False
            _STAR_WINDOWS[key] = win
        if win is False:
            continue
        cw, cf, h, cl, cb = win
        x = ((whole - cw) + frac - cf) / h
        if -1.0 <= x <= 1.0:
            return float(ev(x, cl)) % 360.0, float(ev(x, cb))
    return direct(whole, frac)


def fixstar(name, jd_et, flags=_c.FLG_SWIEPH):
    """Position of a fixed star at Ephemeris Time. Returns ((lon, lat, dist, ...), star name, flags)."""
    flags = int(flags)
    try:
        st = stars.find(_star_key(name))
    except KeyError:
        raise Error(f"star {name!r} not found")
    whole, frac = timescales.split_jd(float(jd_et))
    whole, frac = float(whole), float(frac)
    kw = _observer_kwargs(flags & ~(_c.FLG_TOPOCTR | _c.FLG_HELCTR | _c.FLG_BARYCTR))

    def direct(w, f):
        return stars.apparent(_spk(), st, w, f, deflection=kw["deflection"], aberration=kw["aberration"],
                              nutation=kw["nutation"], frame=kw["frame"], equatorial=kw["equatorial"])
    lon, lat = _star_window(st.index, flags & ~_c.FLG_SIDEREAL, whole, frac, direct)
    lon, lat = float(lon), float(lat)
    if flags & _c.FLG_SIDEREAL:
        lon -= _sidereal_ayanamsa(whole + frac, flags)
    return (lon % 360.0, lat, 1.0, 0.0, 0.0, 0.0), st.name or str(st.hip), flags


def fixstar_ut(name, jd_ut, flags=_c.FLG_SWIEPH):
    return fixstar(name, _tt_from_ut(jd_ut), flags)


fixstar2 = fixstar
fixstar2_ut = fixstar_ut


# ---------------------------------------------------------------------------------------------
# coordinate transforms
# ---------------------------------------------------------------------------------------------
def cotrans(xpo, eps):
    """Rotate (lon, lat, dist) about the x axis by `eps` degrees: ecliptic to equator for positive eps."""
    lo, la, r = np.radians(xpo[0]), np.radians(xpo[1]), xpo[2]
    e = np.radians(eps)
    x, y, z = np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)
    y2 = y * np.cos(e) - z * np.sin(e)
    z2 = y * np.sin(e) + z * np.cos(e)
    return (float(np.degrees(np.arctan2(y2, x)) % 360.0), float(np.degrees(np.arcsin(np.clip(z2, -1, 1)))), float(r))


def cotrans_sp(xpo, eps):
    lon, lat, r = cotrans(xpo[:3], eps)
    return (lon, lat, r, xpo[3], xpo[4], xpo[5])


def azalt(tjd_ut, calc_flag, geopos, atpress, attemp, xin):
    """Azimuth and altitude. Azimuth is measured from the south point, westward.

    calc_flag ECL2HOR: xin = (ecliptic lon, lat); EQU2HOR: xin = (RA, Dec). geopos = (lon, lat, height).
    Returns (azimuth, true altitude, apparent altitude), degrees.
    """
    tt = _tt_from_ut(tjd_ut)
    lst = ((math.degrees(_gast_rad(tjd_ut, tt)) + float(geopos[0])) % 360.0) + _st_offset_deg(tjd_ut)
    if calc_flag == _c.ECL2HOR:
        eps = math.degrees(_true_obliquity_rad(tt))
        ra, dec, _ = cotrans((xin[0], xin[1], 1.0), eps)
    else:
        ra, dec = xin[0], xin[1]
    ha = np.radians(lst - ra)
    lat, d = np.radians(geopos[1]), np.radians(dec)
    sin_alt = np.sin(lat) * np.sin(d) + np.cos(lat) * np.cos(d) * np.cos(ha)
    alt = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    az = np.arctan2(np.sin(ha), np.cos(ha) * np.sin(lat) - np.tan(d) * np.cos(lat))   # from the south, westward
    az = float(np.degrees(az) % 360.0)
    true_alt = float(np.degrees(alt))
    app = true_alt
    if atpress and atpress > 0:
        app = true_alt + _refraction(true_alt, atpress, attemp)
    return az, true_alt, app


def azalt_rev(tjd_ut, calc_flag, geopos, xin):
    """Inverse of azalt (no refraction): from azimuth (south, westward) and true altitude to ecliptic or equatorial."""
    tt = _tt_from_ut(tjd_ut)
    lst = ((math.degrees(_gast_rad(tjd_ut, tt)) + float(geopos[0])) % 360.0) + _st_offset_deg(tjd_ut)
    az, alt = np.radians(xin[0]), np.radians(xin[1])
    lat = np.radians(geopos[1])
    sin_d = np.sin(lat) * np.sin(alt) - np.cos(lat) * np.cos(alt) * np.cos(az)
    d = np.arcsin(np.clip(sin_d, -1.0, 1.0))
    ha = np.arctan2(np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt) * np.sin(lat) + np.sin(alt) * np.cos(lat))
    ra = (lst - np.degrees(ha)) % 360.0
    if calc_flag == _c.HOR2ECL:
        eps = math.degrees(_true_obliquity_rad(tt))
        lon, lat_, _ = cotrans((ra, float(np.degrees(d)), 1.0), -eps)
        return lon, lat_
    return float(ra), float(np.degrees(d))


def _refraction(true_alt, pressure, temp):
    """Atmospheric refraction in degrees (Saemundsson's formula, scaled for pressure and temperature)."""
    h = max(true_alt, -1.5)
    r = 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11))) / 60.0
    return float(r * (pressure / 1010.0) * (283.0 / (273.0 + temp)))


# ---------------------------------------------------------------------------------------------
# eclipses
# ---------------------------------------------------------------------------------------------
def _ut_from_tt(tt):
    return tt - (deltat(tt) if _S.delta_t_user is not None else float(timescales.delta_t(tt)) / 86400.0)


def lun_eclipse_when(jd_start, ifl=_c.FLG_SWIEPH, ifltype=0, backward=False):
    """Next (or previous, if backward) lunar eclipse after a UT Julian Day. Returns (flags, tret) with UT times:
    tret[0] maximum, [2] and [3] partial begin and end, [4] and [5] totality begin and end,
    [6] and [7] penumbral begin and end."""
    from . import eclipses
    ecl = eclipses.next_lunar_eclipse(_spk(), _tt_from_ut(jd_start), bool(backward))
    t = [0.0] * 10
    t[0] = _ut_from_tt(ecl.maximum)
    if ecl.partial:
        t[2], t[3] = _ut_from_tt(ecl.partial[0]), _ut_from_tt(ecl.partial[1])
    if ecl.total:
        t[4], t[5] = _ut_from_tt(ecl.total[0]), _ut_from_tt(ecl.total[1])
    t[6], t[7] = _ut_from_tt(ecl.penumbral[0]), _ut_from_tt(ecl.penumbral[1])
    return ecl.kind, tuple(t)


def sol_eclipse_when_glob(jd_start, ifl=_c.FLG_SWIEPH, ifltype=0, backward=False):
    """Next (or previous) solar eclipse anywhere on Earth. tret[0] is the time of greatest eclipse, [2] and [3] the
    beginning and end of the eclipse on Earth (UT)."""
    from . import eclipses
    ecl = eclipses.next_solar_eclipse(_spk(), _tt_from_ut(jd_start), bool(backward))
    t = [0.0] * 10
    t[0] = _ut_from_tt(ecl.maximum)
    t[2], t[3] = _ut_from_tt(ecl.begin), _ut_from_tt(ecl.end)
    return ecl.kind, tuple(t)


def sol_eclipse_when_loc(jd_start, geopos, ifl=_c.FLG_SWIEPH, backward=False):
    """Next (or previous) solar eclipse at a place (longitude, latitude, height). tret[0] is the maximum, [1] to [4]
    the first to fourth contacts (UT). Returns (flags, tret, attr)."""
    from . import eclipses
    dt = deltat(jd_start)
    ecl = eclipses.next_local_solar_eclipse(_spk(), _tt_from_ut(jd_start), tuple(geopos), dt, bool(backward))
    t = [0.0] * 10
    t[0] = _ut_from_tt(ecl.maximum)
    t[1] = _ut_from_tt(ecl.first)
    t[2] = _ut_from_tt(ecl.second) if ecl.second else 0.0
    t[3] = _ut_from_tt(ecl.third) if ecl.third else 0.0
    t[4] = _ut_from_tt(ecl.fourth)
    attr = [0.0] * 20
    attr[0] = ecl.magnitude
    return ecl.kind, tuple(t), tuple(attr)


# ---------------------------------------------------------------------------------------------
# rise, set, transit
# ---------------------------------------------------------------------------------------------
def rise_trans(tjdut, body, rsmi, geopos, atpress=0.0, attemp=0.0, flags=_c.FLG_SWIEPH):
    """Next rising, setting or meridian transit after a UT Julian Day. Returns (0, (time,)) when found, or
    (-2, (0.0,)) when the body does not rise or set (circumpolar)."""
    from . import risetrans
    spk = _spk()
    if isinstance(body, str):
        try:
            target = stars.find(_star_key(body))
        except KeyError:
            raise Error(f"star {body!r} not found")
    elif body in _PLANET_ID_CENTRES:
        target = _PLANET_ID_CENTRES[body]
    else:
        raise Error(f"body {body} is not available for rise and set")
    kind = ("rise" if rsmi & _c.CALC_RISE else "set" if rsmi & _c.CALC_SET else
            "upper transit" if rsmi & _c.CALC_MTRANSIT else "lower transit" if rsmi & _c.CALC_ITRANSIT else None)
    if kind is None:
        raise Error("rsmi must include CALC_RISE, CALC_SET, CALC_MTRANSIT or CALC_ITRANSIT")
    twilight = ("civil" if rsmi & _c.BIT_CIVIL_TWILIGHT else "nautical" if rsmi & _c.BIT_NAUTIC_TWILIGHT else
                "astronomical" if rsmi & _c.BIT_ASTRO_TWILIGHT else None)
    hindu = (rsmi & _c.BIT_HINDU_RISING) == _c.BIT_HINDU_RISING
    limb = "center" if (rsmi & _c.BIT_DISC_CENTER or hindu) else "bottom" if rsmi & _c.BIT_DISC_BOTTOM else "upper"
    t = risetrans.next_event(spk, target, float(tjdut), tuple(geopos), kind, deltat(tjdut), limb=limb,
                             refraction=not (rsmi & _c.BIT_NO_REFRACTION or hindu), pressure=atpress,
                             temperature=attemp, twilight=twilight, geocentric=hindu)
    if t is None:
        return -2, (0.0,) * 10
    return 0, (t,) + (0.0,) * 9


# ---------------------------------------------------------------------------------------------
# phenomena
# ---------------------------------------------------------------------------------------------
def pheno(jd_et, ipl, iflag=_c.FLG_SWIEPH):
    """Phase angle, illuminated fraction, elongation, apparent disc diameter (degrees) and magnitude of a planet.
    Returns a 20-tuple whose first five entries are those quantities; the Moon's magnitude is NaN."""
    from . import phenomena
    if ipl not in _PLANET_ID_CENTRES or _PLANET_ID_CENTRES[ipl] == planets.EARTH:
        raise Error(f"phenomena are not available for body {ipl}")
    a = phenomena.phenomena(_spk(), _PLANET_ID_CENTRES[ipl], float(jd_et))
    return tuple(a) + (0.0,) * 15


def pheno_ut(jd_ut, ipl, iflag=_c.FLG_SWIEPH):
    return pheno(_tt_from_ut(jd_ut), ipl, iflag)


# ---------------------------------------------------------------------------------------------
# error type: callers written for the Swiss interface catch swe.Error, so convert our own exceptions
# ---------------------------------------------------------------------------------------------
def _wrap_errors(fn):
    import functools

    @functools.wraps(fn)
    def inner(*a, **k):
        try:
            return fn(*a, **k)
        except Error:
            raise
        except (ValueError, KeyError, IndexError, RuntimeError, ArithmeticError, FloatingPointError) as exc:
            raise Error(str(exc)) from exc
    return inner


for _name in ("calc", "calc_ut", "fixstar", "fixstar_ut", "fixstar2", "fixstar2_ut", "houses", "houses_ex",
              "houses_armc", "lun_eclipse_when", "sol_eclipse_when_glob", "sol_eclipse_when_loc", "rise_trans",
              "pheno", "pheno_ut", "azalt", "azalt_rev", "get_ayanamsa_ex", "get_ayanamsa_ex_ut", "get_ayanamsa",
              "get_ayanamsa_ut", "sidtime", "sidtime0", "time_equ"):
    globals()[_name] = _wrap_errors(globals()[_name])


# more functions live in swe_extras (utilities, crossings, refraction, bookkeeping)
from .swe_extras import *  # noqa: E402,F401,F403


if os.environ.get("ALIF_EPHEMERIS_EXACT", "") not in ("1", "true", "yes"):
    set_reference_convention(True)
