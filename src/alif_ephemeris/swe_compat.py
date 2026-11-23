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
    folders += [os.path.join(root, "data", "raw"), os.path.join(root, "data", "packs"), os.getcwd()]
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
        for name in ("alif_past.bsp", "alif_future.bsp"):       # compact packs cut from DE441 (tools/make_range_pack.py)
            f = os.path.join(folder, name)
            if os.path.isfile(f) and norm(f) not in seen:
                seen.add(norm(f))
                found.append(f)
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
    _drop_windows()
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
    _drop_windows()


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
    y, m = int(year), int(month)
    if m <= 2:
        y -= 1
        m += 12
    b = 0
    if gregflag == _c.GREG_CAL:
        a = y // 100
        b = 2 - a + a // 4
    return float(math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + float(day) + b - 1524.5) + float(hour) / 24.0


def revjul(jd, gregflag=_c.GREG_CAL):
    jd = float(jd) + 0.5
    z = math.floor(jd)
    f = jd - z
    if gregflag == _c.GREG_CAL:
        alpha = math.floor((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - alpha // 4
    else:
        a = z
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day = b - d - math.floor(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    whole = math.floor(day)
    return int(year), int(month), int(whole), (day - whole) * 24.0


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
    last = _EE_LAST[0]                          # one tuple, replaced whole, so threads never see a mixed pair
    if last is not None and last[0] == tt:
        return last[1]
    out = _ee_deps_uncached(tt)
    _EE_LAST[0] = (tt, out)
    return out


_EE_LAST = [None]


_EEP: dict = {}


def _ee_deps_uncached(tt):
    """(equation of the equinoxes, nutation in obliquity) in radians from a 4 day polynomial window (degree 11,
    agrees with the full series to about 1e-5 arcsec)."""
    k = math.floor((tt - 2451545.0) / 4.0)
    win = _EEP.get(k)
    if win is None:
        if len(_EEP) > 20000:
            _EEP.clear()
        h = 2.0 + _SUB_MARGIN
        centre = 2451545.0 + (k + 0.5) * 4.0
        c_whole = math.floor(centre)
        c_frac = centre - c_whole
        m = 16
        un = np.cos(np.pi * (np.arange(m) + 0.5) / m)
        times = c_whole + c_frac + un * h
        ee = np.asarray(sidereal_time.equation_of_equinoxes(times), dtype=float)
        deps = np.asarray(frames.nutation_iau2000a(times)[1], dtype=float)
        cheb = np.polynomial.chebyshev
        pe = cheb.cheb2poly(cheb.chebfit(un, ee, 11)).tolist()
        pd = cheb.cheb2poly(cheb.chebfit(un, deps, 11)).tolist()
        win = _EEP[k] = (c_whole, c_frac, h, pe[::-1], pd[::-1])
    c_whole, c_frac, h, pe, pd = win
    u = ((tt - c_whole) - c_frac) / h
    a = b = 0.0
    for x in pe:
        a = a * u + x
    for x in pd:
        b = b * u + x
    return a, b


def _gast_rad(jd_ut, tt):
    """Greenwich apparent sidereal time (radians) using the windowed equation of the equinoxes."""
    jd = float(jd_ut)                                           # Greenwich sidereal time with plain arithmetic
    tt = float(tt)
    whole = math.floor(jd - 0.5) + 0.5
    frac = jd - whole
    f = whole % 1.0 + frac % 1.0
    era = 6.283185307179586 * (f + 0.7790572732640 + 0.00273781191135448 * ((whole - 2451545.0) + frac))
    t = (tt - 2451545.0) / 36525.0
    poly = (0.014506 + t * (4612.156534 + t * (1.3915817 + t * (-0.00000044 + t * (-0.000029956 + t * -0.0000000368)))))         * 4.84813681109536e-06
    return (era + poly + _ee_deps(tt)[0]) % 6.283185307179586


def _true_obliquity_rad(tt):
    t = (float(tt) - 2451545.0) / 36525.0
    mean = (84381.406 + t * (-46.836769 + t * (-0.0001831 + t * (0.00200340 + t * (-0.000000576 + t * (-0.0000000434))))))         * 4.84813681109536e-06
    return mean + _ee_deps(tt)[1]


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
    return jd_lmt + time_equ(jd_ut)


def lat_to_lmt(jd_lat, geolon):
    jd_lmt = jd_lat
    for _ in range(4):
        jd_lmt = jd_lat - time_equ(jd_lmt - geolon / 360.0)
    return jd_lmt


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
    30: "Galactic Centre (Gil Brand)", 31: "Galactic equator (IAU 1958)", 32: "Galactic equator (true)",
    33: "Galactic equator (Mula)", 39: "True Sheoran (Delta Cancri)",
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
    if np.ndim(jd_et) == 0:
        return flags | _c.FLG_SWIEPH, _scalar_ayanamsa(float(jd_et), flags)[0] % 360.0
    ay = _mean_ayanamsa(jd_et)
    if not (flags & _c.FLG_NONUT):
        ay += float(np.degrees(frames.nutation_iau2000a(jd_et)[0]))
    return flags | _c.FLG_SWIEPH, ay % 360.0


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
    _drop_windows()


_SMALL_BODY = {_c.CHIRON: _sb.CHIRON, _c.PHOLUS: _sb.PHOLUS, _c.CERES: _sb.CERES, _c.PALLAS: _sb.PALLAS,
               _c.JUNO: _sb.JUNO, _c.VESTA: _sb.VESTA}

_NAMES = {0: "Sun", 1: "Moon", 2: "Mercury", 3: "Venus", 4: "Mars", 5: "Jupiter", 6: "Saturn", 7: "Uranus",
          8: "Neptune", 9: "Pluto", 10: "mean Node", 11: "true Node", 12: "mean Apogee", 13: "osc. Apogee",
          14: "Earth", 15: "Chiron", 16: "Pholus", 17: "Ceres", 18: "Pallas", 19: "Juno", 20: "Vesta",
          21: "intp. Apogee", 22: "intp. Perigee"}


def _drop_windows():
    """Forget every fitted window: they depend on the data files, the outer-planet convention and the sidereal mode."""
    _WINDOWS.clear()
    _SUBWIN.clear()
    _SUBHITS.clear()
    _STARSUB.clear()
    _AYWIN.clear()
    _STAR_WINDOWS.clear()


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


# ---------------------------------------------------------------------------------------------
# scalar fast path: short polynomial windows evaluated with plain Python arithmetic
# ---------------------------------------------------------------------------------------------
# (days per window, polynomial degree). Each window is fitted to the validated long windows above and checked
# against them at extra points; where it does not reach 0.0003 arcsec the older path is used.
_SUBSPAN = {_c.SUN: (8.0, 11), _c.MOON: (1.0, 13), _c.MERCURY: (4.0, 11), _c.VENUS: (8.0, 11), _c.MARS: (8.0, 11),
            _c.JUPITER: (16.0, 11), _c.SATURN: (16.0, 11), _c.URANUS: (16.0, 11), _c.NEPTUNE: (16.0, 11),
            _c.PLUTO: (16.0, 11), _c.MEAN_NODE: (8.0, 9), _c.MEAN_APOG: (8.0, 11), _c.TRUE_NODE: (2.0, 11)}
_SUBWIN: dict = {}
_SUBHITS: dict = {}
_SUB_HOT = 64                     # calls into one window before a short polynomial window is fitted for it (about 1.4 ms to fit)
_SUBWIN_MAX = 30000
_SCALAR_EXCLUDE = _c.FLG_XYZ | _c.FLG_RADIANS | _c.FLG_TOPOCTR
_SUB_MARGIN = 0.002
_SUB_CHECK = np.array([-0.97, -0.62, -0.2, 0.15, 0.58, 0.9, 1.0])


def _poly_window(fn, span, degree, k, ncols, fallback_fn=None):
    """Fit fn(whole, frac_array) -> (ncols, m) over window k of `span` days with a polynomial in u = (t - centre) / h.
    Returns (c_whole, c_frac, h, [power coefficients per column], [derivative coefficients per column]) or False."""
    h = span / 2.0 + _SUB_MARGIN
    centre = 2451545.0 + (k + 0.5) * span
    c_whole = math.floor(centre)
    c_frac = centre - c_whole
    m = degree + 4
    un = np.cos(np.pi * (np.arange(m) + 0.5) / m)
    both = np.concatenate([un, _SUB_CHECK])
    vals = np.asarray(fn(float(c_whole), c_frac + both * h), dtype=float).reshape(ncols, -1)
    cheb = np.polynomial.chebyshev
    coefs, ders, worst = [], [], 0.0
    for i in range(ncols):
        y = vals[i]
        if i == 0:
            y = np.degrees(np.unwrap(np.radians(y)))
        c = cheb.chebfit(un, y[:m], degree)
        fit = cheb.chebval(_SUB_CHECK, c)
        worst = max(worst, float(np.abs(fit - y[m:]).max()) * (3600.0 if i < 2 else 0.0))
        pw = cheb.cheb2poly(c)
        coefs.append(pw.tolist())
        ders.append((pw[1:] * np.arange(1, len(pw))).tolist())
    if worst > 3e-4:
        if fallback_fn is not None:                 # the long windows are not smooth enough here: use the full calculation
            return _poly_window(fallback_fn, span, degree, k, ncols)
        return False
    return (c_whole, c_frac, h, coefs, ders)


def _lld_fn(body, base, exact=False):
    def fn(whole, fracs):
        lon, lat, dist = (_planet_lonlat_dist(body, whole, fracs, base, whole + fracs) if exact
                          else _fast_lld(body, whole, fracs, base, None))
        return np.stack([np.asarray(lon, dtype=float), np.asarray(lat, dtype=float), np.asarray(dist, dtype=float)])
    return fn


def _scalar_calc(jd_tt, body, flags):
    """(lon, lat, dist, dlon, dlat, ddist) for one time from a short polynomial window, or None to use the long path."""
    spec = _SUBSPAN.get(body)
    if spec is None:
        return None
    span0, degree = spec
    base = flags & ~(_c.FLG_SPEED | _c.FLG_SPEED3 | _c.FLG_SIDEREAL)
    files = _data_files()
    win = False
    for li in (0, 1, 2):                                   # the usual window, then halves where it does not fit
        span = span0 / (1 << li)
        k = math.floor((jd_tt - 2451545.0) / span)
        key = (body, base, files, li, k)
        win = _SUBWIN.get(key)
        if win is None:
            # A window costs a few milliseconds to fit, so it only pays for places that are asked about repeatedly. A
            # sparse sweep (one call every few days) is served by the long windows instead.
            hits = _SUBHITS.get(key, 0) + 1
            if hits < _SUB_HOT:
                if len(_SUBHITS) > 200000:
                    _SUBHITS.clear()
                _SUBHITS[key] = hits
                return None
            if len(_SUBWIN) >= _SUBWIN_MAX:
                _SUBWIN.clear()
            win = _SUBWIN[key] = _poly_window(_lld_fn(body, base), span, degree, k, 3, _lld_fn(body, base, True))
        if win is not False:
            break
    if win is False:
        return None
    c_whole, c_frac, h, coefs, ders = win
    u = ((jd_tt - c_whole) - c_frac) / h
    if u > 1.0 or u < -1.0:
        return None
    v = []
    for c in coefs:
        a = 0.0
        for x in reversed(c):
            a = a * u + x
        v.append(a)
    lon = v[0]
    d = (0.0, 0.0, 0.0)
    if flags & (_c.FLG_SPEED | _c.FLG_SPEED3):
        d = []
        for c in ders:
            a = 0.0
            for x in reversed(c):
                a = a * u + x
            d.append(a / h)
    if flags & _c.FLG_SIDEREAL:
        want = bool(flags & (_c.FLG_SPEED | _c.FLG_SPEED3))
        ay, ay_rate = _scalar_ayanamsa(jd_tt, flags, want)
        lon -= ay
        if want:
            d[0] -= ay_rate
    return (lon % 360.0, v[1], v[2], d[0], d[1], d[2])


_AYWIN: dict = {}


def _scalar_ayanamsa(jd_tt, flags, want_rate=False):
    """(ayanamsa in degrees, its rate per day). The mean part comes from a 32 day polynomial window (degree 7); the
    nutation part, when wanted, from the short nutation windows."""
    nonut = bool(flags & _c.FLG_NONUT)
    mode = (_S.sid_mode, _S.sid_t0, _S.sid_ayan_t0, bool(_CONVENTION["outer_barycentres"]))
    win = False
    for li, span in enumerate((32.0, 16.0, 8.0)):          # the usual window, then shorter ones where it is not exact enough
        k = math.floor((jd_tt - 2451545.0) / span)
        key = (mode, li, k)
        win = _AYWIN.get(key)
        if win is None:
            if len(_AYWIN) > 20000:
                _AYWIN.clear()

            def fn(whole, fracs):
                return np.array([[_mean_ayanamsa(whole + float(f)) for f in fracs]])
            win = _AYWIN[key] = _poly_window_plain(fn, span, 7, k)
        if win is not False:
            break
    if win is False:                                          # not smooth enough here: the exact value, slower
        a = _mean_ayanamsa(jd_tt)
        r = (_mean_ayanamsa(jd_tt + 0.005) - _mean_ayanamsa(jd_tt - 0.005)) / 0.01 if want_rate else 0.0
        if not nonut:
            a += math.degrees(frames._nutation_windowed(jd_tt)[0])
            if want_rate:
                r += math.degrees(frames._nutation_windowed(jd_tt + 0.005)[0] - frames._nutation_windowed(jd_tt - 0.005)[0]) / 0.01
        return a, r
    c_whole, c_frac, h, c, dc = win
    u = ((jd_tt - c_whole) - c_frac) / h
    a = r = 0.0
    for x in reversed(c):
        a = a * u + x
    if want_rate:
        for x in reversed(dc):
            r = r * u + x
        r /= h
    if not nonut:
        nk = math.floor((jd_tt - 2451545.0) / 2.0)
        nw = _AYWIN.get(("nut", nk))
        if nw is None:
            def nfn(whole, fracs):
                return np.array([[frames._nutation_windowed(whole + float(f))[0] * 57.29577951308232 for f in fracs]])
            nw = _AYWIN[("nut", nk)] = _poly_window_plain(nfn, 2.0, 8, nk)
        if nw is False:
            a += math.degrees(frames._nutation_windowed(jd_tt)[0])
            if want_rate:
                r += math.degrees(frames._nutation_windowed(jd_tt + 0.005)[0] - frames._nutation_windowed(jd_tt - 0.005)[0]) / 0.01
            return a, r
        c_whole, c_frac, h, c, dc = nw
        u = ((jd_tt - c_whole) - c_frac) / h
        n = 0.0
        for x in reversed(c):
            n = n * u + x
        a += n
        if want_rate:
            m = 0.0
            for x in reversed(dc):
                m = m * u + x
            r += m / h
    return a, r


def _poly_window_plain(fn, span, degree, k):
    """One-column polynomial window; False when it does not reproduce the function to 0.0003 arcsec at extra points."""
    h = span / 2.0 + _SUB_MARGIN
    centre = 2451545.0 + (k + 0.5) * span
    c_whole = math.floor(centre)
    c_frac = centre - c_whole
    m = degree + 4
    un = np.cos(np.pi * (np.arange(m) + 0.5) / m)
    both = np.concatenate([un, _SUB_CHECK])
    y = np.asarray(fn(float(c_whole), c_frac + both * h), dtype=float).reshape(-1)
    cheb = np.polynomial.chebyshev
    c = cheb.chebfit(un, y[:m], degree)
    if float(np.abs(cheb.chebval(_SUB_CHECK, c) - y[m:]).max()) * 3600.0 > 3e-4:
        return False
    pw = cheb.cheb2poly(c)
    return (c_whole, c_frac, h, pw.tolist(), (pw[1:] * np.arange(1, len(pw))).tolist())


_SITE_CACHE: dict = {}


def _scalar_topo(jd_tt, body, flags, jd_ut, topo=None):
    """Topocentric (lon, lat, dist) for one time with plain arithmetic: the fast geocentric position minus the
    observer's offset, plus the diurnal aberration. Matches the full computation to about 0.001 arcsec; the Moon
    (large parallax, second order terms) and speed requests use the full path instead."""
    if topo is None:
        topo = _S.topo
    if topo is None:
        return None
    geo = _scalar_calc(jd_tt, body, flags & ~(_c.FLG_TOPOCTR | _c.FLG_EQUATORIAL))
    if geo is None:
        return None
    site = _SITE_CACHE.get(topo)
    if site is None:
        lat = math.radians(topo[1])
        e2 = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563)
        n = 6378.137 / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
        hh = topo[2] / 1000.0
        site = (( n + hh) * math.cos(lat), (n * (1.0 - e2) + hh) * math.sin(lat), math.radians(topo[0]))
        if len(_SITE_CACHE) > 64:
            _SITE_CACHE.clear()
        _SITE_CACHE[topo] = site
    xy, zz, lon_e = site
    if jd_ut is None:
        jd_ut = jd_tt - deltat(jd_tt)
    theta = (_gast_rad(jd_ut, jd_tt) + lon_e) % 6.283185307179586
    eps = _true_obliquity_rad(jd_tt)
    ce, se = math.cos(eps), math.sin(eps)
    ct, st = math.cos(theta), math.sin(theta)
    x, y, z = xy * ct, xy * st, zz
    sx, sy, sz = x, y * ce + z * se, -y * se + z * ce            # observer, true ecliptic of date (km)
    lo, la = math.radians(geo[0]), math.radians(geo[1])
    d_km = geo[2] * 149597870.7
    cl = math.cos(la)
    vx = d_km * cl * math.cos(lo) - sx
    vy = d_km * cl * math.sin(lo) - sy
    vz = d_km * math.sin(la) - sz
    r = math.sqrt(vx * vx + vy * vy + vz * vz)
    if not flags & (_c.FLG_TRUEPOS | _c.FLG_NOABERR):
        w = 7.2921150e-5 * xy                                    # the observer's own velocity (km/s), turning with the Earth
        wx0, wy0 = -w * st, w * ct
        wx, wy, wz = wx0, wy0 * ce, -wy0 * se
        k = 299792.458
        ux, uy, uz = vx / r, vy / r, vz / r
        dot = (ux * wx + uy * wy + uz * wz) / k
        ux, uy, uz = ux + wx / k - dot * ux, uy + wy / k - dot * uy, uz + wz / k - dot * uz
        nn = math.sqrt(ux * ux + uy * uy + uz * uz)
        vx, vy, vz = ux / nn * r, uy / nn * r, uz / nn * r
    if flags & _c.FLG_EQUATORIAL:
        vy, vz = vy * ce - vz * se, vy * se + vz * ce
    return (math.degrees(math.atan2(vy, vx)) % 360.0, math.degrees(math.asin(max(-1.0, min(1.0, vz / r)))),
            r / 149597870.7)


def _calc_tt(jd_tt, body, flags, jd_ut=None):
    """Cached front door of the position calculation. A position depends on the time, body, flags and the settings
    held in `_S`, so all of them form the key; software that asks for the same place twice gets it back at once."""
    if (flags & _c.FLG_TOPOCTR and body != _c.MOON and body in _SUBSPAN
            and not flags & (_c.FLG_XYZ | _c.FLG_RADIANS | _c.FLG_NONUT | _c.FLG_J2000 | _c.FLG_HELCTR | _c.FLG_BARYCTR
                             | _c.FLG_SPEED | _c.FLG_SPEED3 | _c.FLG_SIDEREAL)):
        r = _scalar_topo(float(jd_tt), body, int(flags), jd_ut)
        if r is not None:
            return (r[0], r[1], r[2], 0.0, 0.0, 0.0), flags
    if not flags & _SCALAR_EXCLUDE and body in _SUBSPAN:
        r = _scalar_calc(float(jd_tt), body, int(flags))
        if r is not None:
            return r, flags
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
_WINDOW_LEVELS = {_c.MOON: [(16.0, 40), (2.0, 12)], _c.SUN: [(64.0, 56), (16.0, 20)], _c.MERCURY: [(16.0, 64), (4.0, 16)],
                  _c.VENUS: [(64.0, 96), (16.0, 32), (4.0, 12)], _c.MARS: [(32.0, 40), (8.0, 14)],
                  _c.JUPITER: [(64.0, 240), (8.0, 20)], _c.SATURN: [(32.0, 40), (8.0, 14)],
                  _c.URANUS: [(128.0, 64), (32.0, 32), (8.0, 16)], _c.NEPTUNE: [(128.0, 64), (32.0, 32), (8.0, 16)],
                  _c.PLUTO: [(128.0, 64), (32.0, 32), (8.0, 16)],
                  _c.MEAN_NODE: [(128.0, 24), (16.0, 16)], _c.MEAN_APOG: [(32.0, 32), (8.0, 14)],
                  _c.TRUE_NODE: [(16.0, 28), (4.0, 12)]}
_WINDOW_MARGIN = 0.03
_KS: dict = {}
_CHECK = np.array([-0.93, -0.41, 0.07, 0.52, 0.95])           # extra points, used to validate a fit


_DCT: dict = {}


def _cheb_interpolate(values):
    """Chebyshev coefficients of the polynomial through values given at the n Chebyshev nodes cos(pi (i + 1/2) / n):
    the same answer as a least squares fit with n coefficients, but one small matrix product."""
    n = values.shape[0]
    basis = _DCT.get(n)
    if basis is None:
        j = np.arange(n)
        basis = _DCT[n] = np.cos(np.pi * np.outer(j, j + 0.5) / n) * (2.0 / n)
    coef = basis @ values
    coef[0] *= 0.5
    return coef


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
    coef = _cheb_interpolate(np.stack([lon[:n], lat[:n], dist[:n]], axis=1))          # one (n, 3) block of coefficients
    fit = ev0(_CHECK, coef)                                                            # (3, 5)
    err = max(np.abs(fit[0] - lon[n:]).max(), np.abs(fit[1] - lat[n:]).max()) * 3600.0
    return (c_whole, c_frac, h, coef) if err < 1e-3 else False


def _fast_topo(body, tt_whole, tt_frac, flags, jd_ut):
    """Topocentric longitude, latitude, distance from the fast geocentric windows: the observer's offset from the
    Earth's centre (in the true ecliptic of date) is subtracted from the geocentric vector. Differs from the full
    computation by well under 0.001 arcsec (checked in tests)."""
    g_flags = flags & ~(_c.FLG_TOPOCTR | _c.FLG_EQUATORIAL)
    lon, lat, dist = _fast_lld(body, tt_whole, tt_frac, g_flags, jd_ut)
    fr = np.asarray(tt_frac, dtype=float)
    tt = float(tt_whole) + fr
    ut = jd_ut if jd_ut is not None else tt - np.asarray(timescales.delta_t(tt)) / 86400.0
    r_eq, v_eq = positions.site_vector_true_equator(_S.topo[0], _S.topo[1], _S.topo[2], ut, tt)
    eps = frames.true_obliquity(tt)
    ce, se = np.cos(eps), np.sin(eps)
    x, y, z = r_eq[..., 0], r_eq[..., 1], r_eq[..., 2]
    sx, sy, sz = x, y * ce + z * se, -y * se + z * ce                       # observer in the true ecliptic of date (km)
    lo, la = np.radians(lon), np.radians(lat)
    d_km = np.asarray(dist) * positions.AU_KM
    vx = d_km * np.cos(la) * np.cos(lo) - sx
    vy = d_km * np.cos(la) * np.sin(lo) - sy
    vz = d_km * np.sin(la) - sz
    r = np.sqrt(vx * vx + vy * vy + vz * vz)
    if not flags & (_c.FLG_TRUEPOS | _c.FLG_NOABERR):                           # the observer's own turning adds a diurnal aberration
        wx, wy, wz = v_eq[..., 0], v_eq[..., 1] * ce + v_eq[..., 2] * se, -v_eq[..., 1] * se + v_eq[..., 2] * ce
        ux, uy, uz = vx / r, vy / r, vz / r
        k = 299792.458
        dot = (ux * wx + uy * wy + uz * wz) / k
        ux, uy, uz = ux + wx / k - dot * ux, uy + wy / k - dot * uy, uz + wz / k - dot * uz
        n = np.sqrt(ux * ux + uy * uy + uz * uz)
        vx, vy, vz = ux / n * r, uy / n * r, uz / n * r
    if flags & _c.FLG_EQUATORIAL:
        vy, vz = vy * ce - vz * se, vy * se + vz * ce
    out_lon = np.degrees(np.arctan2(vy, vx)) % 360.0
    out_lat = np.degrees(np.arcsin(np.clip(vz / r, -1.0, 1.0)))
    return out_lon, out_lat, r / positions.AU_KM


def _fast_lld(body, tt_whole, tt_frac, flags, jd_ut):
    """Same answer as _planet_lonlat_dist, much faster when many nearby times are asked for in turn. A Chebyshev fit
    of the body's own computed position is built once per window of days and reused; each fit is checked against
    extra points computed the full way and used only if it agrees to 0.001 arcsec (tests/test_extras.py)."""
    levels = _WINDOW_LEVELS.get(body)
    if (levels is not None and body != _c.MOON and flags & _c.FLG_TOPOCTR and _S.topo is not None
            and not flags & (_c.FLG_XYZ | _c.FLG_RADIANS | _c.FLG_NONUT | _c.FLG_J2000 | _c.FLG_HELCTR | _c.FLG_BARYCTR)):
        return _fast_topo(body, tt_whole, tt_frac, flags, jd_ut)
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
    # like the reference, a letter that names no system falls back to Placidus
    if hsys in ("I", "i", "J", "Q", "Y"):
        raise Error(f"house system {hsys!r} (APC, Sunshine, Savard-A, Pullen delta) is not built yet in Alif Ephemeris")
    return _houses.LETTERS.get(hsys, "Placidus")


def _refuse_beyond_polar_circle(name, lat, eps):
    """The reference program refuses Placidus and Koch inside the polar circles (latitude 90 degrees minus the
    obliquity), because their definition fails there. We can compute something, but software written for the
    reference expects the error and falls back to another system, so the Swiss-style layer raises it too."""
    if name in ("Placidus", "Koch") and abs(lat) >= 90.0 - eps:
        raise Error(f"{name} houses cannot be computed at latitude {lat:.2f} (inside the polar circle)")


def _extra_angles(ramc, eps, lat):
    """(equatorial ascendant, Koch co-ascendant, Munkasey co-ascendant, polar ascendant) in degrees.
    East point: the ascendant at latitude 0. Koch's co-ascendant: the ascendant at the opposite latitude. Munkasey's:
    the plain ascendant formula at the co-latitude (90 - latitude), with no check for the side of the horizon. Polar
    ascendant: the ascendant half a turn of sidereal time later."""
    equasc = float(_houses.ascendant_from(ramc, eps, 0.0))
    co1 = float(_houses.ascendant_from(ramc, eps, -lat))
    th, e = math.radians(ramc), math.radians(eps)
    colat = math.radians(90.0 - lat)
    tan_c = math.tan(colat) if abs(lat) > 1e-12 else 1e16
    co2 = math.degrees(math.atan2(math.cos(th), -(math.sin(th) * math.cos(e) + tan_c * math.sin(e)))) % 360.0
    pol = float(_houses.ascendant_from(ramc + 180.0, eps, lat))
    return equasc, co1, co2, pol


def _gauquelin_houses(jd_ut, lat, lon, tt, flags):
    ramc = (math.degrees(_gast_rad(jd_ut, tt)) + float(lon)) % 360.0 + _st_offset_deg(jd_ut)
    eps = math.degrees(_true_obliquity_rad(tt))
    try:
        cusps = _houses.gauquelin_sectors(ramc, eps, lat)
    except ValueError as exc:
        raise Error(str(exc))
    h = _houses.compute(ramc, eps, lat, "Porphyry")
    asc, mc, vtx = h.ascendant, h.midheaven, h.vertex
    ascmc = [asc, mc, ramc, vtx, *_extra_angles(ramc, eps, lat)]
    if flags & _c.FLG_SIDEREAL:
        ay = _sidereal_ayanamsa(tt, flags)
        cusps = [(c - ay) % 360.0 for c in cusps]
        for i in (0, 1, 3, 4, 5, 6, 7):
            ascmc[i] = (ascmc[i] - ay) % 360.0
    return tuple(cusps), tuple(ascmc)


def _houses_placidus_py(ramc, eps, lat):
    """Placidus cusps and angles for one place and time with plain arithmetic: (cusps[12], asc, mc, vertex, equasc)
    in degrees, or None when the method fails (inside the polar circles) so the general code takes over."""
    pi, tau = math.pi, 2.0 * math.pi
    sin, cos, tan, atan2, acos, sqrt = math.sin, math.cos, math.tan, math.atan2, math.acos, math.sqrt
    th, e, ph = math.radians(ramc), math.radians(eps), math.radians(lat)
    ste, cte, sth, cth = sin(e), cos(e), sin(th), cos(th)
    tan_phi = tan(ph)
    mc = atan2(sth, cth * cte) % tau
    asc = atan2(cth, -(sth * cte + tan_phi * ste)) % tau
    if (asc - mc) % tau >= pi:
        asc = (asc + pi) % tau
    # vertex: the ascendant seen from co-latitude with the meridian turned half way round
    th2 = th + pi
    cot_phi = 1.0 / tan_phi if tan_phi != 0.0 else 1e16
    vtx = atan2(cos(th2), -(sin(th2) * cte + cot_phi * ste)) % tau
    mc2 = atan2(sin(th2), cos(th2) * cte) % tau
    if (vtx - mc2) % tau >= pi:
        vtx = (vtx + pi) % tau
    equasc = atan2(cth, -sth * cte) % tau
    mc_equasc = mc
    if (equasc - mc_equasc) % tau >= pi:
        equasc = (equasc + pi) % tau
    dm = (asc - mc) % tau
    da = pi - dm
    third = 1.0 / 3.0
    cus = {}
    for h, frac, from_noon, x in ((11, third, True, mc + dm * third), (12, 2.0 * third, True, mc + 2.0 * dm * third),
                                  (2, 2.0 * third, False, asc + da * third), (3, third, False, asc + 2.0 * da * third)):
        base = th if from_noon else th + pi
        off = 0.0 if from_noon else pi                  # from the nadir side the arc is measured back from 180 degrees
        for _ in range(40):
            sd = ste * sin(x)
            t = -tan_phi * sd / sqrt(1.0 - sd * sd)
            if t >= 1.0 or t <= -1.0:
                return None
            ra = base + frac * (acos(t) - off)
            x1 = atan2(sin(ra), cos(ra) * cte) % tau
            d1 = (x1 - x + pi) % tau - pi
            if -1e-9 < d1 < 1e-9:
                x = x1
                break
            sd = ste * sin(x1)
            t = -tan_phi * sd / sqrt(1.0 - sd * sd)
            if t >= 1.0 or t <= -1.0:
                return None
            ra = base + frac * (acos(t) - off)
            x2 = atan2(sin(ra), cos(ra) * cte) % tau
            d2 = (x2 - x1 + pi) % tau - pi
            den = d2 - d1
            x = (x - d1 * d1 / den) % tau if abs(den) > 1e-15 else x2
        cus[h] = x
    deg = math.degrees
    c = [0.0] * 12
    c[0], c[9], c[3], c[6] = deg(asc), deg(mc), deg((mc + pi) % tau), deg((asc + pi) % tau)
    for h, o in ((11, 5), (12, 6), (2, 8), (3, 9)):
        c[h - 1] = deg(cus[h])
        c[o - 1] = deg((cus[h] + pi) % tau)
    return c, c[0], c[9], deg(vtx), deg(equasc)


try:                                                    # the compiled twin (csrc/alif_core.c), when it was built
    from . import _core
    _houses_placidus_fast = _core.placidus
except ImportError:
    _core = None
    _houses_placidus_fast = _houses_placidus_py


def houses_ex(jd_ut, lat, lon, hsys=b"P", flags=0):
    """House cusps and angles. Returns (cusps, ascmc): 12 cusp longitudes and 8 angle values (degrees)."""
    name = _hsys_name(hsys)
    tt = _tt_from_ut(jd_ut)
    if name == "Gauquelin":
        return _gauquelin_houses(jd_ut, lat, lon, tt, flags)
    eps = math.degrees(_true_obliquity_rad(tt))
    _refuse_beyond_polar_circle(name, float(lat), eps)
    ramc = (math.degrees(_gast_rad(jd_ut, tt)) + float(lon)) % 360.0 + _st_offset_deg(jd_ut)
    if flags & _c.FLG_NONUT:
        gm = float(np.degrees(sidereal_time.gmst(jd_ut, tt))) + _st_offset_deg(jd_ut)
        ramc = (gm + lon) % 360.0
        eps = float(np.degrees(frames.mean_obliquity(tt)))
    fast = _houses_placidus_fast(ramc, eps, float(lat)) if name == "Placidus" else None
    if fast is not None:
        cusps, asc, mc, vtx, equasc = fast
    else:
        h = _houses.compute(ramc, eps, lat, name)
        cusps = list(h.cusps)
        asc, mc, vtx = h.ascendant, h.midheaven, h.vertex
        equasc = float(_houses.ascendant_from(ramc, eps, 0.0))
    if name in ("Regiomontanus", "Campanus", "Topocentric"):      # for these the Midheaven is the tenth cusp itself
        mc = cusps[9]
    _eq, co1, co2, pol = _extra_angles(ramc, eps, float(lat))
    ascmc = [asc, mc, ramc, vtx, equasc, co1, co2, pol]
    if flags & _c.FLG_SIDEREAL:
        ay = _scalar_ayanamsa(float(tt), flags)[0]
        cusps = [(c - ay) % 360.0 for c in cusps]
        for i in (0, 1, 3, 4, 5, 6, 7):
            ascmc[i] = (ascmc[i] - ay) % 360.0
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
        return tuple(cusps), (h.ascendant, h.midheaven, armc % 360.0, h.vertex, *_extra_angles(armc, eps, float(lat)))
    _refuse_beyond_polar_circle(name, float(lat), float(eps))
    h = _houses.compute(armc, eps, lat, name)
    mc = h.cusps[9] if name in ("Regiomontanus", "Campanus", "Topocentric") else h.midheaven
    return tuple(h.cusps), (h.ascendant, mc, armc % 360.0, h.vertex, *_extra_angles(armc, eps, float(lat)))


def house_name(hsys):
    return _hsys_name(hsys)


# ---------------------------------------------------------------------------------------------
# fixed stars
# ---------------------------------------------------------------------------------------------
def _star_key(name):
    key = str(name).strip()
    if key.startswith(","):                                   # ",gaCen": the Bayer code alone
        return key.lstrip(",").strip()
    if "," in key:
        head = key.split(",")[0].strip()
        if head:
            return head
    return key.lstrip(",").strip()


_STAR_WINDOWS: dict = {}
_STAR_LEVELS = [(32.0, 48), (8.0, 20)]


_STAR_BY_NAME: dict = {}
_STARSUB: dict = {}


def _scalar_star(index, base, jd_tt, direct):
    """(lon, lat) of a star from a 16 day polynomial window (degree 11) fitted to the full calculation, or None."""
    k = math.floor((jd_tt - 2451545.0) / 16.0)
    key = (index, base, _data_files(), k)
    win = _STARSUB.get(key)
    if win is None:
        if len(_STARSUB) > 30000:
            _STARSUB.clear()
        win = False
        try:
            h = 8.0 + _SUB_MARGIN
            centre = 2451545.0 + (k + 0.5) * 16.0
            cw = math.floor(centre)
            cf = centre - cw
            m = 16
            un = np.cos(np.pi * (np.arange(m) + 0.5) / m)
            both = np.concatenate([un, _SUB_CHECK])
            lo, la = direct(float(cw), cf + both * h)
            lo = np.degrees(np.unwrap(np.radians(np.asarray(lo, dtype=float).reshape(-1))))
            la = np.asarray(la, dtype=float).reshape(-1)
            if lo.size == both.size:
                cheb = np.polynomial.chebyshev
                cl, cb = cheb.chebfit(un, lo[:m], 11), cheb.chebfit(un, la[:m], 11)
                err = max(np.abs(cheb.chebval(_SUB_CHECK, cl) - lo[m:]).max(),
                          np.abs(cheb.chebval(_SUB_CHECK, cb) - la[m:]).max()) * 3600.0
                if err < 3e-4:
                    win = (cw, cf, h, cheb.cheb2poly(cl).tolist()[::-1], cheb.cheb2poly(cb).tolist()[::-1])
        except Exception:
            win = False
        _STARSUB[key] = win
    if win is False:
        return None
    cw, cf, h, pl, pb = win
    u = ((jd_tt - cw) - cf) / h
    if u > 1.0 or u < -1.0:
        return None
    a = b = 0.0
    for x in pl:
        a = a * u + x
    for x in pb:
        b = b * u + x
    return a % 360.0, b


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
    st = _STAR_BY_NAME.get(name)
    if st is None:
        try:
            st = stars.find(_star_key(name))
        except KeyError:
            raise Error(f"star {name!r} not found")
        if isinstance(name, str):
            _STAR_BY_NAME[name] = st
    kw = _observer_kwargs(flags & ~(_c.FLG_TOPOCTR | _c.FLG_HELCTR | _c.FLG_BARYCTR))

    def direct(w, f):
        return stars.apparent(_spk(), st, w, f, deflection=kw["deflection"], aberration=kw["aberration"],
                              nutation=kw["nutation"], frame=kw["frame"], equatorial=kw["equatorial"])
    base = flags & ~_c.FLG_SIDEREAL
    jd = float(jd_et)
    hit = _scalar_star(st.index, base, jd, direct)
    if hit is None:
        whole, frac = timescales.split_jd(jd)
        whole, frac = float(whole), float(frac)
        lon, lat = _star_window(st.index, base, whole, frac, direct)
        hit = (float(lon), float(lat))
    lon, lat = hit
    if flags & _c.FLG_SIDEREAL:
        lon -= _scalar_ayanamsa(jd, flags)[0]
    return (lon % 360.0, lat, 1.0, 0.0, 0.0, 0.0), stars.label(st), flags


def fixstar_ut(name, jd_ut, flags=_c.FLG_SWIEPH):
    return fixstar(name, _tt_from_ut(jd_ut), flags)


fixstar2 = fixstar
fixstar2_ut = fixstar_ut


# ---------------------------------------------------------------------------------------------
# coordinate transforms
# ---------------------------------------------------------------------------------------------
def _rotate_x(xpo, eps):
    """Rotate (lon, lat, dist) about the x axis by `eps` degrees: ecliptic to equator for positive eps."""
    lo, la, r = math.radians(xpo[0]), math.radians(xpo[1]), xpo[2]
    e = math.radians(eps)
    cl = math.cos(la)
    x, y, z = cl * math.cos(lo), cl * math.sin(lo), math.sin(la)
    ce, se = math.cos(e), math.sin(e)
    y2 = y * ce - z * se
    z2 = y * se + z * ce
    return (math.degrees(math.atan2(y2, x)) % 360.0, math.degrees(math.asin(max(-1.0, min(1.0, z2)))), float(r))


def cotrans(xpo, eps):
    """Rotate (lon, lat, dist) between the ecliptic and the equator. As in the reference: a negative `eps` goes from the
    ecliptic to the equator, a positive one from the equator to the ecliptic."""
    return _rotate_x(xpo, -eps)


def cotrans_sp(xpo, eps):
    """Like cotrans, and the speeds are rotated too (longitude, latitude and distance per day)."""
    lon, lat, r = cotrans(xpo[:3], eps)
    h = 1e-4                                                  # speeds by a tiny step along the motion
    lo, la, dr = float(xpo[3]), float(xpo[4]), float(xpo[5])
    a = cotrans((xpo[0] + lo * h, xpo[1] + la * h, xpo[2] + dr * h), eps)
    b = cotrans((xpo[0] - lo * h, xpo[1] - la * h, xpo[2] - dr * h), eps)
    return (lon, lat, r, ((a[0] - b[0] + 180.0) % 360.0 - 180.0) / (2 * h), (a[1] - b[1]) / (2 * h), (a[2] - b[2]) / (2 * h))


def azalt(tjd_ut, calc_flag, geopos, atpress, attemp, xin):
    """Azimuth and altitude. Azimuth is measured from the south point, westward.

    calc_flag ECL2HOR: xin = (ecliptic lon, lat); EQU2HOR: xin = (RA, Dec). geopos = (lon, lat, height).
    Returns (azimuth, true altitude, apparent altitude), degrees.
    """
    tt = _tt_from_ut(tjd_ut)
    lst = ((math.degrees(_gast_rad(tjd_ut, tt)) + float(geopos[0])) % 360.0) + _st_offset_deg(tjd_ut)
    if calc_flag == _c.ECL2HOR:
        eps = math.degrees(_true_obliquity_rad(tt))
        ra, dec, _ = _rotate_x((xin[0], xin[1], 1.0), eps)
    else:
        ra, dec = xin[0], xin[1]
    ha = math.radians(lst - ra)
    lat, d = math.radians(geopos[1]), math.radians(dec)
    sin_lat, cos_lat, sin_d, cos_d = math.sin(lat), math.cos(lat), math.sin(d), math.cos(d)
    cos_ha = math.cos(ha)
    sin_alt = sin_lat * sin_d + cos_lat * cos_d * cos_ha
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    az = math.atan2(math.sin(ha), cos_ha * sin_lat - math.tan(d) * cos_lat)   # from the south, westward
    az = math.degrees(az) % 360.0
    true_alt = math.degrees(alt)
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
        lon, lat_, _ = _rotate_x((ra, float(np.degrees(d)), 1.0), -eps)
        return lon, lat_
    return float(ra), float(np.degrees(d))


def _refraction(true_alt, pressure, temp):
    """Atmospheric refraction in degrees (Saemundsson's formula, scaled for pressure and temperature)."""
    h = max(true_alt, -1.5)
    r = 1.02 / math.tan(math.radians(h + 10.3 / (h + 5.11))) / 60.0
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


def _local_noon_near(t_ut, t_tt):
    """The moment the Moon's shadow lies where it is local apparent noon (the Sun on the meridian there), or 0.0."""
    from . import swe_extras
    from scipy.optimize import brentq

    def meridian_side(t):                                   # zero when the Sun is on the meridian there, above or below the pole
        lon = swe_extras.sol_eclipse_where(t, _c.FLG_SWIEPH)[1][0]
        tt = _tt_from_ut(t)
        sun = calc(tt, _c.SUN, _c.FLG_SWIEPH | _c.FLG_EQUATORIAL)[0]
        return math.sin(math.radians(math.degrees(_gast_rad(t, tt)) + lon - sun[0]))
    try:
        lo, hi = t_ut - 0.12, t_ut + 0.12
        grid = [lo + (hi - lo) * i / 16.0 for i in range(17)]
        vals = [meridian_side(t) for t in grid]
        roots = []
        for a, b, fa, fb in zip(grid, grid[1:], vals, vals[1:]):
            if fa * fb <= 0.0:
                roots.append(float(brentq(meridian_side, a, b, xtol=1e-8)))
        if roots:
            return min(roots, key=lambda t: abs(t - t_ut))
    except Exception:
        pass
    return 0.0


def sol_eclipse_when_glob(jd_start, ifl=_c.FLG_SWIEPH, ifltype=0, backward=False):
    """Next (or previous) solar eclipse anywhere on Earth. tret[0] is the time of greatest eclipse, [2] and [3] the
    beginning and end of the eclipse on Earth (UT)."""
    from . import eclipses
    ecl = eclipses.next_solar_eclipse(_spk(), _tt_from_ut(jd_start), bool(backward))
    t = [0.0] * 10
    t[0] = _ut_from_tt(ecl.maximum)
    t[2], t[3] = _ut_from_tt(ecl.begin), _ut_from_tt(ecl.end)
    t[1] = _local_noon_near(t[0], ecl.maximum)
    central = eclipses.solar_central_times(_spk(), ecl.maximum) if ecl.kind & _c.ECL_CENTRAL else None
    if central:
        (t[4], t[5]), (t[6], t[7]) = ((_ut_from_tt(a), _ut_from_tt(b)) for a, b in central)
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
    from . import swe_extras
    site = tuple(geopos)
    flag = ecl.kind
    seen = {0: _c.ECL_MAX_VISIBLE, 1: _c.ECL_1ST_VISIBLE, 2: _c.ECL_2ND_VISIBLE, 3: _c.ECL_3RD_VISIBLE, 4: _c.ECL_4TH_VISIBLE}
    for slot, bit in seen.items():                               # a contact counts as visible when the Sun is up then
        if t[slot] and eclipses._sun_altitude(_spk(), _tt_from_ut(t[slot]), site, t[slot]) > -0.5:
            flag |= bit | _c.ECL_VISIBLE
    attr = list(swe_extras.sol_eclipse_how(t[0], site, ifl)[1])
    attr[0] = ecl.magnitude if not attr[0] else attr[0]
    return flag, tuple(t), tuple(float(x) for x in attr)


# ---------------------------------------------------------------------------------------------
# rise, set, transit
# ---------------------------------------------------------------------------------------------
_FAST_RISE = True
_REV_CENTRES: dict = {}


def _fast_lst(ut, tt, east_lon_deg):
    return (_gast_rad(ut, tt) + math.radians(east_lon_deg)) % 6.283185307179586


def _fast_rise_equatorial(body, tt, site, ut):
    """(RA, Dec in radians, distance in AU) of a planet seen from `site`, for the rise and set search. Plain arithmetic on
    the fast windows; the Moon is within 0.3 arcsec of the full computation, the others within 0.001 arcsec."""
    rev = _REV_CENTRES.get(_CONVENTION["outer_barycentres"])
    if rev is None:
        rev = _REV_CENTRES[_CONVENTION["outer_barycentres"]] = {v: k for k, v in _PLANET_ID_CENTRES.items()}
    swe_body = rev.get(body)
    if swe_body is None or swe_body not in _SUBSPAN:
        return None
    r = _scalar_topo(float(tt), swe_body, _c.FLG_SWIEPH | _c.FLG_EQUATORIAL | _c.FLG_TOPOCTR, float(ut), tuple(site))
    if r is None:
        return None
    return math.radians(r[0]), math.radians(r[1]), r[2]


def rise_trans(tjdut, body, rsmi, geopos, atpress=0.0, attemp=0.0, flags=_c.FLG_SWIEPH):
    """Next rising, setting or meridian transit after a UT Julian Day. Returns (0, (time,)) when found, or
    (-2, (0.0,)) when the body does not rise or set (circumpolar)."""
    from . import risetrans
    risetrans.FAST_EQUATORIAL = _fast_rise_equatorial if _FAST_RISE else None
    risetrans.FAST_LST = _fast_lst if _FAST_RISE else None
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
