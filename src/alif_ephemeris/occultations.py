"""Occultations: the Moon passing in front of a planet or a star.

Same shadow geometry as the solar eclipses (eclipses.py). A star is a point at infinity, so the Moon's shadow on
the Earth is a cylinder as wide as the Moon. A planet is a small disc at a finite distance, so its shadow narrows
(umbra) or widens (penumbra) with distance by the planet's radius. Angles in degrees, distances in km.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import eclipses, planets, positions, stars, timescales

PLANET_RADIUS_KM = {planets.MERCURY: 2440.0, planets.VENUS: 6052.0, planets.MARS: 3396.0, planets.JUPITER: 71492.0,
                    planets.SATURN: 60268.0, planets.URANUS: 25559.0, planets.NEPTUNE: 24764.0, planets.PLUTO: 1188.0,
                    planets.SUN: 696000.0}


def _geocentric(spk, body, tt, **kw):
    w, f = timescales.split_jd(float(tt))
    return np.asarray(positions.observe(spk, body, float(w), float(f), cartesian=True, equatorial=True, **kw)) * planets.AU_KM


def _star_direction(spk, star, tt):
    w, f = timescales.split_jd(float(tt))
    ra, dec = stars.apparent(spk, star, float(w), float(f), equatorial=True)
    ra, dec = np.radians(float(ra)), np.radians(float(dec))
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


@dataclass
class Target:
    """A planet (NAIF-style id from planets.py) or a star (stars.Star)."""
    body: object

    @property
    def is_star(self):
        return isinstance(self.body, stars.Star)

    def radius(self):
        return 0.0 if self.is_star else PLANET_RADIUS_KM.get(self.body, 0.0)

    def vector(self, spk, tt, **kw):
        """Geocentric (or topocentric) vector in km; stars are a very distant point in their direction."""
        if self.is_star:
            return _star_direction(spk, self.body, tt) * 1.0e15
        return _geocentric(spk, self.body, tt, **kw)


def _moon(spk, tt, **kw):
    return _geocentric(spk, planets.MOON, tt, **kw)


def _axis_geometry(spk, target, tt):
    """(distance of the Earth's centre from the shadow axis, penumbra radius, umbra radius, gap of the axis from the
    Earth's silhouette, axis unit vector, Moon-to-Earth-centre distance along the axis)."""
    rm = _moon(spk, tt)
    rt = target.vector(spk, tt)
    ax = rm - rt
    length = float(np.linalg.norm(ax))
    a = ax / length
    c = rm - float(rm @ a) * a                                 # point of the axis nearest the Earth's centre, from the centre
    dist = float(np.linalg.norm(c))
    s = float(-rm @ a)                                         # distance from the Moon along the axis to the centre's plane
    if target.is_star:
        pen = umb = eclipses.R_MOON_SOLAR
    else:
        spread = target.radius() * s / length
        pen, umb = eclipses.R_MOON_SOLAR + spread, eclipses.R_MOON_SOLAR - spread
    pole = np.array([0.0, 0.0, 1.0])
    yh = pole - (pole @ a) * a
    yh = yh / np.linalg.norm(yh)
    xh = np.cross(yh, a)
    r_p = eclipses.R_EARTH_SOLAR * (1.0 - eclipses.EARTH_FLATTENING)
    cosd2 = 1.0 - (a @ pole) ** 2
    b = np.sqrt(r_p ** 2 * cosd2 + eclipses.R_EARTH_SOLAR ** 2 * (1.0 - cosd2))
    gap = eclipses._ellipse_gap(float(c @ xh), float(c @ yh), eclipses.R_EARTH_SOLAR, float(b))
    return dist, pen, umb, gap, a, s


@dataclass
class GlobalOccultation:
    maximum: float
    central: bool
    partial: tuple          # begin, end of the shadow touching the Earth
    umbral: tuple           # begin, end of the full-width shadow touching the Earth
    centre_line: tuple      # begin, end of the axis touching the Earth
    min_gap: float


def _conjunction_candidates(spk, target, tt_start, backwards, max_days=7500.0):
    """Times (TT) near each conjunction in ecliptic longitude of the Moon with the target within reach of an occultation."""
    step = 1.0 * (-1.0 if backwards else 1.0)          # the Moon gains 13 degrees a day on a planet or star: one crossing per step at most

    def sep_lon(tt):
        w, f = timescales.split_jd(float(tt))
        lon_m, lat_m, _ = positions.observe(spk, planets.MOON, float(w), float(f))
        if target.is_star:
            lon_t, lat_t = stars.apparent(spk, target.body, float(w), float(f))
            lon_t, lat_t = np.radians(float(lon_t)), np.radians(float(lat_t))
        else:
            lon_t, lat_t, _ = positions.observe(spk, target.body, float(w), float(f))
        return (float(lon_m) - float(lon_t) + np.pi) % (2 * np.pi) - np.pi, float(lat_m) - float(lat_t)

    t = float(tt_start)
    d0, b0 = sep_lon(t)
    n = int(max_days / abs(step))
    for _ in range(n):
        t1 = t + step
        d1, b1 = sep_lon(t1)
        crossed = (d0 <= 0.0 < d1) or (d0 >= 0.0 > d1)
        if crossed and abs(d1 - d0) < 1.0:
            lo, hi = (t, t1) if t < t1 else (t1, t)
            tc = brentq(lambda x: sep_lon(x)[0], lo, hi, xtol=1e-6)
            if abs(np.degrees(sep_lon(tc)[1])) < 1.9:           # Moon radius + parallax + planet size
                yield tc
        t, d0, b0 = t1, d1, b1


def next_global_occultation(spk, target, tt_start, backwards=False):
    for tc in _conjunction_candidates(spk, target, tt_start, backwards):
        tmax = eclipses._minimize(lambda t: _axis_geometry(spk, target, t)[0], tc - 0.3, tc + 0.3)
        dist, pen, umb, gap, a, s = _axis_geometry(spk, target, tmax)
        if gap >= pen:
            continue
        if (tmax > tt_start) != (not backwards):
            continue

        def edges(level):
            g = lambda t: _axis_geometry(spk, target, t)[3] - level(_axis_geometry(spk, target, t))
            return brentq(g, tmax - 0.3, tmax, xtol=1e-9), brentq(g, tmax, tmax + 0.3, xtol=1e-9)
        partial = edges(lambda v: v[1])
        umbral = edges(lambda v: abs(v[2])) if gap < abs(umb) else partial
        centre = edges(lambda v: 0.0) if gap < 0 else (tmax, tmax)
        return GlobalOccultation(tmax, gap < 0, partial, umbral, centre, gap)
    return None


@dataclass
class LocalOccultation:
    maximum: float
    first: float
    second: float | None
    third: float | None
    fourth: float
    magnitude: float
    min_separation_deg: float


def _local_separation(spk, target, tt, site, jd_ut):
    """(separation of the centres, Moon angular radius, target angular radius) in radians, seen from a site."""
    rm = _moon(spk, tt, observer="site", site=site, jd_ut=jd_ut)
    rt = target.vector(spk, tt, observer="site", site=site, jd_ut=jd_ut) if not target.is_star else target.vector(spk, tt)
    sep = float(np.arctan2(np.linalg.norm(np.cross(rm, rt)), rm @ rt))
    r_m = float(np.arcsin(eclipses.R_MOON_SOLAR / np.linalg.norm(rm)))
    r_t = 0.0 if target.is_star else float(np.arcsin(min(1.0, target.radius() / np.linalg.norm(rt))))
    return sep, r_m, r_t


def next_local_occultation(spk, target, tt_start, site, delta_t_days, backwards=False):
    for tc in _conjunction_candidates(spk, target, tt_start, backwards):
        ut = lambda t: t - delta_t_days
        f = lambda t: _local_separation(spk, target, t, site, ut(t))[0]
        tmax = eclipses._minimize(f, tc - 0.12, tc + 0.12)
        sep, r_m, r_t = _local_separation(spk, target, tmax, site, ut(tmax))
        if sep >= r_m + r_t:
            continue
        if (tmax > tt_start) != (not backwards):
            continue

        def contact(level):
            g = lambda t: _local_separation(spk, target, t, site, ut(t))[0] - level(*_local_separation(spk, target, t, site, ut(t))[1:])
            return brentq(g, tmax - 0.12, tmax, xtol=1e-9), brentq(g, tmax, tmax + 0.12, xtol=1e-9)
        first, fourth = contact(lambda rm_, rt_: rm_ + rt_)
        total = sep < r_m - r_t
        second = third = None
        if total:
            second, third = contact(lambda rm_, rt_: rm_ - rt_)
        mag = (r_m + r_t - sep) / (2 * r_t) if r_t > 0 else 1.0
        return LocalOccultation(tmax, first, second, third, fourth, min(mag, 1.0) if total else mag, float(np.degrees(sep)))
    return None
