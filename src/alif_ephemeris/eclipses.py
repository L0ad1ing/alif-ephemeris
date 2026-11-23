"""Solar and lunar eclipses: search, type, and the times of the contacts.

Geometry (standard shadow-cone model):
  * Lunar: the Earth's shadow is a cone behind the Earth. The Moon is eclipsed while its centre is within the
    shadow radius (plus or minus the Moon's radius) of the shadow axis, at the Moon's distance.
  * Solar, whole Earth: the Moon's shadow axis is the line through the Sun's and Moon's centres. The eclipse
    exists somewhere on Earth while that axis passes within (Earth radius + penumbra radius) of the centre.
  * Solar, one place: the Sun and Moon as seen from the observer overlap while their centres are closer than
    the sum of their angular radii.

All times are Julian Days in Terrestrial Time unless a function says UT. The body radii are module constants.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from . import frames, planets, positions, timescales

# Radii in km. The Sun and Earth values are the usual conventional ones; see docs for the calibration.
R_SUN = 696000.0
R_EARTH = 6378.14
R_MOON = 1738.5                           # calibrated: contact times of 16 reference lunar eclipses, rms 1.6 s
UMBRA_ENLARGEMENT = 1.0101                 # factors on the Earth's radius inside the two shadow cones
PENUMBRA_ENLARGEMENT = 1.0101

SYNODIC = 29.530588853

# flag bits of the interface (same values as the Swiss-style constants)
ECL_CENTRAL, ECL_NONCENTRAL, ECL_TOTAL, ECL_ANNULAR, ECL_PARTIAL, ECL_ANNULAR_TOTAL, ECL_PENUMBRAL = 1, 2, 4, 8, 16, 32, 64


def _geocentric(spk, body, tt, **kw):
    """Geocentric vector (km) of a body at Terrestrial Time `tt`."""
    w, f = timescales.split_jd(float(tt))
    return np.asarray(positions.observe(spk, body, float(w), float(f), cartesian=True, equatorial=True, **kw)) * planets.AU_KM


_TRUE = dict(light_time=False, deflection=False, aberration=False)


def _sun_moon(spk, tt):
    """Geocentric Sun and Moon vectors, apparent: with light time, light bending and aberration.

    Using the apparent places for both bodies reproduces the published circumstances of eclipses; the shadow
    axis is built from the places where the two bodies are seen."""
    return _geocentric(spk, planets.SUN, tt), _geocentric(spk, planets.MOON, tt)


def _elong(spk, tt):
    """Signed ecliptic-longitude elongation of the Moon from the Sun in degrees, wrapped to (-180, 180]."""
    w, f = timescales.split_jd(float(tt))
    ls = positions.observe(spk, planets.SUN, float(w), float(f), **_TRUE)[0]
    lm = positions.observe(spk, planets.MOON, float(w), float(f), **_TRUE)[0]
    return (np.degrees(lm - ls) + 180.0) % 360.0 - 180.0


def _minimize(f, lo, hi, n=48):
    """Time of the minimum of f on [lo, hi]: coarse grid first, then Brent's method around the best point."""
    ts = np.linspace(lo, hi, n)
    vals = np.array([f(t) for t in ts])
    i = int(np.argmin(vals))
    a, b = ts[max(i - 1, 0)], ts[min(i + 1, n - 1)]
    res = minimize_scalar(f, bracket=(a, ts[i], b), method="brent", tol=1e-12) if 0 < i < n - 1 else None
    return float(res.x) if res is not None else float(ts[i])


def _syzygies(spk, tt_start, direction, kind):
    """Yield successive new moons (kind 'new') or full moons ('full') starting after (direction +1) or before
    (direction -1) tt_start."""
    target = 0.0 if kind == "new" else 180.0

    def f(t):
        return (_elong(spk, t) - target + 180.0) % 360.0 - 180.0

    t = tt_start
    while True:
        # step in 3-day increments until the sign of f changes from negative to positive
        step = 3.0 * direction
        a, fa = t, f(t)
        for _ in range(40):
            b = a + step
            fb = f(b)
            if (fa < 0 <= fb and direction > 0) or (fb < 0 <= fa and direction < 0):
                lo, hi = (a, b) if a < b else (b, a)
                root = brentq(f, lo, hi, xtol=1e-9)
                yield root
                t = root + direction * 20.0
                break
            a, fa = b, fb
        else:
            raise RuntimeError("no syzygy found")


# -------------------------------------------------------------------------------------------------
# lunar
# -------------------------------------------------------------------------------------------------
def _lunar_metrics(spk, tt):
    rs, rm = _sun_moon(spk, tt)
    D = np.linalg.norm(rs)
    u = -rs / D
    z = float(rm @ u)
    d = float(np.linalg.norm(rm - z * u))
    re_u, re_p = R_EARTH * UMBRA_ENLARGEMENT, R_EARTH * PENUMBRA_ENLARGEMENT
    rho_u = re_u - z * (R_SUN - re_u) / D
    rho_p = re_p + z * (R_SUN + re_p) / D
    return d, rho_u, rho_p


@dataclass
class LunarEclipse:
    maximum: float
    kind: int
    penumbral: tuple
    partial: tuple | None
    total: tuple | None
    umbral_magnitude: float
    penumbral_magnitude: float


def _lunar_at_full_moon(spk, t_full):
    tmax = _minimize(lambda t: _lunar_metrics(spk, t)[0], t_full - 0.25, t_full + 0.25)
    d, ru, rp = _lunar_metrics(spk, tmax)
    if d >= rp + R_MOON:
        return None
    um = (ru + R_MOON - d) / (2 * R_MOON)
    pm = (rp + R_MOON - d) / (2 * R_MOON)

    def contacts(radius_fn):
        g = lambda t: _lunar_metrics(spk, t)[0] - radius_fn(*_lunar_metrics(spk, t)[1:])
        a = brentq(g, tmax - 0.2, tmax, xtol=1e-9)
        b = brentq(g, tmax, tmax + 0.2, xtol=1e-9)
        return a, b

    pen = contacts(lambda ru_, rp_: rp_ + R_MOON)
    part = contacts(lambda ru_, rp_: ru_ + R_MOON) if d < ru + R_MOON else None
    total = contacts(lambda ru_, rp_: ru_ - R_MOON) if d < ru - R_MOON else None
    kind = ECL_TOTAL if total else ECL_PARTIAL if part else ECL_PENUMBRAL
    return LunarEclipse(tmax, kind, pen, part, total, um, pm)


def next_lunar_eclipse(spk, tt_start, backwards=False):
    direction = -1 if backwards else 1
    for t_full in _syzygies(spk, tt_start, direction, "full"):
        w, f = timescales.split_jd(t_full)
        lat = positions.observe(spk, planets.MOON, float(w), float(f), **_TRUE)[1]
        if abs(np.degrees(lat)) > 1.7:
            continue
        ecl = _lunar_at_full_moon(spk, t_full)
        if ecl is None:
            continue
        if (ecl.maximum > tt_start) == (not backwards):
            return ecl


# -------------------------------------------------------------------------------------------------
# solar, whole Earth
# -------------------------------------------------------------------------------------------------
EARTH_FLATTENING = 1.0 / 298.257223563
R_MOON_SOLAR = 1737.4                      # solar eclipses: calibrated against 20 reference eclipses (begin and end of the whole-Earth
R_EARTH_SOLAR = 6371.0                     # eclipse agree to about 8 seconds rms; only the sum of the two radii is well determined)


def _ellipse_gap(x, y, a, b):
    """Distance from the point (x, y) to the ellipse with semi-axes a (along x) and b (along y); negative inside."""
    th0 = np.arctan2(y / b, x / a)
    f = lambda th: (x - a * np.cos(th)) ** 2 + (y - b * np.sin(th)) ** 2
    res = minimize_scalar(f, bounds=(th0 - 0.6, th0 + 0.6), method="bounded", options={"xatol": 1e-12})
    d = float(np.sqrt(res.fun))
    inside = (x / a) ** 2 + (y / b) ** 2 < 1.0
    return -d if inside else d


def _solar_axis(spk, tt):
    """Moon's shadow axis in the fundamental plane (through the Earth's centre, perpendicular to the axis).

    Returns (distance from the Earth's centre to the axis, penumbra radius, umbra radius, gap between the axis and
    the Earth's silhouette). The silhouette of the flattened Earth seen along the axis is an ellipse. A positive
    umbra radius means the umbra reaches the plane (total possible); negative means annular.
    """
    rs, rm = _sun_moon(spk, tt)
    ax = rm - rs
    L = np.linalg.norm(ax)
    a = ax / L
    c = rs - (rs @ a) * a                                  # point of the axis nearest the Earth's centre
    dist = float(np.linalg.norm(c))
    s = float(-rm @ a)
    pen = R_MOON_SOLAR + s * (R_SUN + R_MOON_SOLAR) / L
    umb = R_MOON_SOLAR - s * (R_SUN - R_MOON_SOLAR) / L
    pole = np.array([0.0, 0.0, 1.0])                       # equatorial frame: the Earth's axis
    yh = pole - (pole @ a) * a
    yh = yh / np.linalg.norm(yh)
    xh = np.cross(yh, a)
    r_p = R_EARTH_SOLAR * (1.0 - EARTH_FLATTENING)
    cosd2 = 1.0 - (a @ pole) ** 2                           # cos^2 of the axis's declination
    b = np.sqrt(r_p ** 2 * cosd2 + R_EARTH_SOLAR ** 2 * (1.0 - cosd2))
    gap = _ellipse_gap(float(c @ xh), float(c @ yh), R_EARTH_SOLAR, float(b))
    return dist, pen, umb, gap


@dataclass
class SolarEclipse:
    maximum: float
    kind: int
    gamma: float
    begin: float
    end: float
    umbra_radius: float


def next_solar_eclipse(spk, tt_start, backwards=False):
    direction = -1 if backwards else 1
    for t_new in _syzygies(spk, tt_start, direction, "new"):
        w, f = timescales.split_jd(t_new)
        lat = positions.observe(spk, planets.MOON, float(w), float(f), **_TRUE)[1]
        if abs(np.degrees(lat)) > 1.7:
            continue
        tmax = _minimize(lambda t: _solar_axis(spk, t)[0], t_new - 0.3, t_new + 0.3)
        dist, pen, umb, gap = _solar_axis(spk, tmax)
        if gap >= pen:
            continue
        g = lambda t: _solar_axis(spk, t)[3] - _solar_axis(spk, t)[1]
        begin = brentq(g, tmax - 0.3, tmax, xtol=1e-9)
        end = brentq(g, tmax, tmax + 0.3, xtol=1e-9)
        if gap < 0:
            kind = (ECL_TOTAL if umb > 0 else ECL_ANNULAR) | ECL_CENTRAL
            if gap > -abs(umb):
                kind = (ECL_TOTAL if umb > 0 else ECL_ANNULAR) | ECL_NONCENTRAL
        elif gap < abs(umb):
            kind = (ECL_TOTAL if umb > 0 else ECL_ANNULAR) | ECL_NONCENTRAL
        else:
            kind = ECL_PARTIAL | ECL_NONCENTRAL
        ecl = SolarEclipse(tmax, kind, dist / R_EARTH, begin, end, umb)
        if (ecl.maximum > tt_start) == (not backwards):
            return ecl


# -------------------------------------------------------------------------------------------------
# solar, one place
# -------------------------------------------------------------------------------------------------
def _local_sep(spk, tt, site, jd_ut):
    w, f = timescales.split_jd(float(tt))
    kw = dict(observer="site", site=site, jd_ut=jd_ut, cartesian=True, equatorial=True)
    s = np.asarray(positions.observe(spk, planets.SUN, float(w), float(f), **kw)) * planets.AU_KM
    m = np.asarray(positions.observe(spk, planets.MOON, float(w), float(f), **kw)) * planets.AU_KM
    sep = float(np.arctan2(np.linalg.norm(np.cross(s, m)), s @ m))
    rs = float(np.arcsin(R_SUN / np.linalg.norm(s)))
    rm = float(np.arcsin(R_MOON / np.linalg.norm(m)))
    return sep, rs, rm


@dataclass
class LocalSolarEclipse:
    maximum: float
    first: float
    second: float | None
    third: float | None
    fourth: float
    kind: int
    magnitude: float


def _sun_altitude(spk, tt, site, jd_ut):
    """Altitude of the Sun's centre in degrees (no refraction)."""
    from . import sidereal_time
    w, f = timescales.split_jd(float(tt))
    ra, dec, _ = positions.observe(spk, planets.SUN, float(w), float(f), observer="site", site=site, jd_ut=jd_ut,
                                   equatorial=True)
    lst = sidereal_time.local_apparent_sidereal_time(jd_ut, site[0], tt)
    ha = lst - ra
    lat = np.radians(site[1])
    return float(np.degrees(np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha))))


def _sun_up_between(spk, t1, t4, site, ut):
    """True if the Sun's centre is above the horizon (less 50 arcmin for refraction and size) at some moment."""
    for t in np.linspace(t1, t4, 13):
        if _sun_altitude(spk, t, site, ut(t)) > -0.8333:
            return True
    return False


def next_local_solar_eclipse(spk, tt_start, site, delta_t_days, backwards=False):
    direction = -1 if backwards else 1

    def ut(t):
        return t - delta_t_days

    for t_new in _syzygies(spk, tt_start, direction, "new"):
        tmax = _minimize(lambda t: _local_sep(spk, t, site, ut(t))[0], t_new - 0.3, t_new + 0.3)
        sep, rs, rm = _local_sep(spk, tmax, site, ut(tmax))
        if sep >= rs + rm:
            continue
        g1 = lambda t: _local_sep(spk, t, site, ut(t))[0] - sum(_local_sep(spk, t, site, ut(t))[1:])
        c1 = brentq(g1, tmax - 0.25, tmax, xtol=1e-9)
        c4 = brentq(g1, tmax, tmax + 0.25, xtol=1e-9)
        if not _sun_up_between(spk, c1, c4, site, ut):
            continue
        c2 = c3 = None
        if sep < abs(rs - rm):
            g2 = lambda t: _local_sep(spk, t, site, ut(t))[0] - abs(_local_sep(spk, t, site, ut(t))[1] - _local_sep(spk, t, site, ut(t))[2])
            c2 = brentq(g2, c1, tmax, xtol=1e-9)
            c3 = brentq(g2, tmax, c4, xtol=1e-9)
        kind = (ECL_TOTAL if rm > rs else ECL_ANNULAR) if c2 else ECL_PARTIAL
        mag = (rs + rm - sep) / (2 * rs)
        ecl = LocalSolarEclipse(tmax, c1, c2, c3, c4, kind, mag)
        if (ecl.maximum > tt_start) == (not backwards):
            return ecl


# -------------------------------------------------------------------------------------------------
# quick existence checks, used to walk back along a Saros series
# -------------------------------------------------------------------------------------------------
SAROS_REACH = 0.9985                       # how far into the Earth's disc the penumbra must reach to count as a series member (calibrated on 160 eclipses)


def _syzygy_near(spk, tt_guess, target):
    """The new moon (target 0) or full moon (target 180) within two days of a guessed time, or None."""
    f = lambda t: (_elong(spk, t) - target + 180.0) % 360.0 - 180.0
    a, b = tt_guess - 2.0, tt_guess + 2.0
    fa, fb = f(a), f(b)
    if not (fa < 0.0 <= fb):
        return None
    return float(brentq(f, a, b, xtol=1e-7))


def solar_eclipse_near(spk, tt_guess):
    """Greatest-eclipse time (TT) of a solar eclipse at the new moon nearest a guessed time, or None if there is none.
    Cheaper than next_solar_eclipse: it only decides whether the shadow reaches the Earth."""
    t_new = _syzygy_near(spk, tt_guess, 0.0)
    if t_new is None:
        return None
    w, fr = timescales.split_jd(t_new)
    if abs(np.degrees(positions.observe(spk, planets.MOON, float(w), float(fr), **_TRUE)[1])) > 1.7:
        return None

    def reach(t):
        rs, rm = _sun_moon(spk, t)
        ax = rm - rs
        L = np.linalg.norm(ax)
        a = ax / L
        c = rs - (rs @ a) * a
        s = float(-rm @ a)
        pen = R_MOON_SOLAR + s * (R_SUN + R_MOON_SOLAR) / L
        return float(np.linalg.norm(c)) - pen
    res = minimize_scalar(reach, bounds=(t_new - 0.3, t_new + 0.3), method="bounded", options={"xatol": 1e-4})
    return float(res.x) if res.fun < R_EARTH_SOLAR * SAROS_REACH else None


def lunar_eclipse_near(spk, tt_guess):
    """Greatest-eclipse time (TT) of a lunar eclipse at the full moon nearest a guessed time, or None."""
    t_full = _syzygy_near(spk, tt_guess, 180.0)
    if t_full is None:
        return None
    w, fr = timescales.split_jd(t_full)
    if abs(np.degrees(positions.observe(spk, planets.MOON, float(w), float(fr), **_TRUE)[1])) > 1.7:
        return None
    res = minimize_scalar(lambda t: _lunar_metrics(spk, t)[0] - _lunar_metrics(spk, t)[2], bounds=(t_full - 0.25, t_full + 0.25),
                          method="bounded", options={"xatol": 1e-4})
    return float(res.x) if res.fun < R_MOON else None


def solar_central_times(spk, tmax):
    """For a central eclipse: (begin, end) of the umbral (or antumbral) phase on Earth, when the umbra's edge first and
    last touches the silhouette, and (begin, end) of the central line, when the axis itself first and last meets the
    Earth. All TT; None when the eclipse is not central or a root is not found."""
    def gap(t):
        return _solar_axis(spk, t)[3]

    def umb_edge(t):
        d, pen, umb, g = _solar_axis(spk, t)
        return g - abs(umb)
    out = []
    for fn in (umb_edge, gap):
        try:
            if fn(tmax) >= 0.0:
                return None
            out.append((float(brentq(fn, tmax - 0.3, tmax, xtol=1e-9)), float(brentq(fn, tmax, tmax + 0.3, xtol=1e-9))))
        except ValueError:
            return None
    return out[0], out[1]
