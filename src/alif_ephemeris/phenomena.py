"""Planetary phenomena: phase angle, illuminated fraction, elongation, apparent size and apparent magnitude.

Geometry comes from the apparent places of the body, the Sun and the Earth. Magnitudes follow Mallama and Hilton (2018),
"Computing Apparent Planetary Magnitudes for The Astronomical Almanac", the formulas the Almanac itself uses. The Moon's
magnitude is not given: it depends on the surface and on the Earthshine in ways no short formula captures.
"""
from __future__ import annotations

import numpy as np

from . import planets, positions, timescales

# volumetric mean diameters in km, used for the apparent size of the disc
DIAMETER_KM = {planets.SUN: 1392000.0, planets.MOON: 3474.8, planets.MERCURY: 4879.4, planets.VENUS: 12103.6,
               planets.MARS: 6779.0, planets.JUPITER: 139822.0, planets.SATURN: 116464.0, planets.URANUS: 50724.0,
               planets.NEPTUNE: 49244.0, planets.PLUTO: 2376.6}

# north pole directions (ICRS right ascension, declination in degrees) for the ring tilt of Saturn and Uranus' latitude
_POLE = {planets.SATURN: (40.589, 83.537), planets.URANUS: (257.311, -15.175)}


def _unit_pole(body):
    ra, dec = np.radians(_POLE[body][0]), np.radians(_POLE[body][1])
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _vec(spk, body, whole, frac, **kw):
    return np.asarray(positions.observe(spk, body, whole, frac, cartesian=True, equatorial=True, **kw))


def phenomena(spk, body, jd_tt):
    """(phase angle deg, illuminated fraction, elongation deg, apparent diameter deg, apparent magnitude)."""
    whole, frac = (float(x) for x in timescales.split_jd(float(jd_tt)))
    d_vec = _vec(spk, body, whole, frac)                                    # Earth to body, apparent
    delta = float(np.linalg.norm(d_vec))
    s_vec = _vec(spk, planets.SUN, whole, frac)                             # Earth to Sun
    sun_dist = float(np.linalg.norm(s_vec))
    diameter = float(np.degrees(2 * np.arcsin(min(1.0, DIAMETER_KM[body] / 2 / (delta * planets.AU_KM)))))
    if body == planets.SUN:
        return 0.0, 1.0, 0.0, diameter, -26.74
    elongation = float(np.degrees(np.arccos(np.clip(d_vec @ s_vec / (delta * sun_dist), -1.0, 1.0))))
    sun_from_body = s_vec - d_vec
    r = float(np.linalg.norm(sun_from_body))
    i = float(np.degrees(np.arccos(np.clip((-d_vec) @ sun_from_body / (delta * r), -1.0, 1.0))))
    phase = (1.0 + np.cos(np.radians(i))) / 2.0
    return i, float(phase), elongation, diameter, _magnitude(spk, body, r, delta, i, whole, frac)


def _year(whole, frac):
    return 2000.0 + (whole + frac - 2451545.0) / 365.25


def _magnitude(spk, body, r, delta, a, whole, frac):
    """Apparent magnitude: r is the Sun distance, delta the Earth distance (AU), a the phase angle (degrees)."""
    lg = 5.0 * np.log10(r * delta)
    if body == planets.MERCURY:
        return -0.613 + 6.3280e-2 * a - 1.6336e-3 * a ** 2 + 3.3644e-5 * a ** 3 - 3.4265e-7 * a ** 4 \
            + 1.6893e-9 * a ** 5 - 3.0334e-12 * a ** 6 + lg
    if body == planets.VENUS:
        if a < 163.7:
            return -4.384 - 1.044e-3 * a + 3.687e-4 * a ** 2 - 2.814e-6 * a ** 3 + 8.938e-9 * a ** 4 + lg
        return 236.05828 - 2.81914 * a + 8.39034e-3 * a ** 2 + lg
    if body == planets.MARS:
        return -1.601 + 0.02267 * a - 0.0001302 * a ** 2 + lg
    if body == planets.JUPITER:
        return -9.395 - 3.7e-4 * a + 6.16e-4 * a ** 2 + lg
    if body == planets.SATURN:
        to_earth = -_vec(spk, body, whole, frac, frame="j2000")
        sin_b = abs(float(_unit_pole(body) @ (to_earth / np.linalg.norm(to_earth))))
        return -8.914 - 1.825 * sin_b + 0.026 * a - 0.378 * sin_b * np.exp(-2.25 * a) + lg
    if body == planets.URANUS:
        to_earth = -_vec(spk, body, whole, frac, frame="j2000")
        lat = np.degrees(np.arcsin(float(_unit_pole(body) @ (to_earth / np.linalg.norm(to_earth)))))
        return -7.110 - 8.4e-4 * abs(lat) + 6.587e-3 * a + 1.045e-4 * a ** 2 + lg
    if body == planets.NEPTUNE:
        y = _year(whole, frac)
        base = -6.89 if y <= 1980.0 else -7.00 if y >= 2000.0 else -6.89 - 0.11 * (y - 1980.0) / 20.0
        return base + lg
    if body == planets.PLUTO:
        return -1.00 + lg
    return float("nan")
