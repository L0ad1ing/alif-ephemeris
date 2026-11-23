"""Rising, setting and meridian transit times.

An event is when the body's chosen edge reaches the horizon (the upper limb by default), allowing for the
atmosphere's refraction, or when it crosses the local meridian. Positions are topocentric and apparent, so the
Moon's parallax is included. The search runs forward from the start time and scans in 20 minute steps for the next
change of sign, then refines it.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from . import frames, planets, positions, sidereal_time, stars, timescales

RADIUS_KM = {planets.SUN: 696000.0, planets.MOON: 1737.4, planets.MERCURY: 2440.0, planets.VENUS: 6052.0,
             planets.MARS: 3396.0, planets.JUPITER: 71492.0, planets.SATURN: 60268.0, planets.URANUS: 25559.0,
             planets.NEPTUNE: 24764.0, planets.PLUTO: 1188.0}

STANDARD_REFRACTION = 34.0 / 60.0           # degrees at the horizon, 1013.25 mbar and 15 C
TWILIGHT = {"civil": -6.0, "nautical": -12.0, "astronomical": -18.0}


HORIZON_REFRACTION = 0.6090                 # degrees: calibrated against reference rise and set times of Sun, Moon, planets and stars


def horizon_refraction(pressure=0.0, temperature=0.0):
    """Refraction at the horizon in degrees; scaled for pressure (hPa) and temperature (C) when given."""
    if pressure <= 0:
        return HORIZON_REFRACTION
    return HORIZON_REFRACTION * (pressure / 1013.25) * (288.15 / (273.15 + temperature))


def depression_for_apparent(semi_diameter, limb, pressure=0.0, temperature=0.0, refraction=True):
    """True altitude of the body's centre (negative degrees) when its chosen limb is on the horizon."""
    offset = semi_diameter if limb == "upper" else -semi_diameter if limb == "bottom" else 0.0
    return -offset - (horizon_refraction(pressure, temperature) if refraction else 0.0)


def _equatorial(spk, body, tt, site, ut, geocentric=False):
    """(RA, Dec) in radians and distance in AU, topocentric apparent for a body, geocentric for a star."""
    w, f = timescales.split_jd(float(tt))
    if isinstance(body, stars.Star):
        ra, dec = stars.apparent(spk, body, float(w), float(f), equatorial=True)
        return np.radians(float(ra)), np.radians(float(dec)), float("inf")
    if geocentric:
        ra, dec, dist = positions.observe(spk, body, float(w), float(f), equatorial=True)
    else:
        ra, dec, dist = positions.observe(spk, body, float(w), float(f), observer="site", site=site, jd_ut=ut, equatorial=True)
    return float(ra), float(dec), float(dist)


def _altitude_and_ha(spk, body, ut, site, delta_t, geocentric=False):
    tt = ut + delta_t
    ra, dec, dist = _equatorial(spk, body, tt, site, ut, geocentric)
    lst = float(sidereal_time.local_apparent_sidereal_time(ut, site[0], tt))
    ha = lst - ra
    lat = np.radians(site[1])
    alt = np.degrees(np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)))
    return float(alt), float((np.degrees(ha) + 180.0) % 360.0 - 180.0), dist


def _semi_diameter_deg(body, dist_au):
    if isinstance(body, stars.Star) or body not in RADIUS_KM or not np.isfinite(dist_au):
        return 0.0
    return float(np.degrees(np.arcsin(RADIUS_KM[body] / (dist_au * planets.AU_KM))))


def next_event(spk, body, ut_start, site, kind, delta_t_days, *, limb="upper", refraction=True, pressure=0.0,
               temperature=0.0, twilight=None, max_days=1.2, geocentric=False):
    """Next event after `ut_start`: kind is "rise", "set", "upper transit" or "lower transit". Returns the UT Julian
    Day, or None if the body does not do it within `max_days` (circumpolar or never rising)."""
    def threshold(ut):
        if twilight:
            return TWILIGHT[twilight]
        if kind not in ("rise", "set"):
            return 0.0
        _, _, dist = _altitude_and_ha(spk, body, ut, site, delta_t_days, geocentric)
        return depression_for_apparent(_semi_diameter_deg(body, dist), limb, pressure, temperature, refraction)

    def g(ut):
        alt, ha, _ = _altitude_and_ha(spk, body, ut, site, delta_t_days, geocentric)
        if kind in ("rise", "set"):
            return alt - threshold(ut)
        if kind == "upper transit":
            return ha
        return (ha + 360.0) % 360.0 - 180.0          # lower transit: hour angle 180 maps to 0

    step = 20.0 / 1440.0
    t0 = ut_start
    v0 = g(t0)
    t = t0
    while t < ut_start + max_days:
        t1 = t + step
        v1 = g(t1)
        up = v0 < 0.0 <= v1
        down = v0 >= 0.0 > v1
        if kind == "rise" and up or kind == "set" and down or kind in ("upper transit", "lower transit") and up and abs(v1 - v0) < 90.0:
            return float(brentq(g, t, t1, xtol=1e-10))
        t, v0 = t1, v1
    return None
