"""Eclipse attributes at one moment and place: how much of the Sun or Moon is covered, where it stands in the sky.

Built on the same shadow geometry as eclipses.py. Angles in degrees, distances in km.
"""
from __future__ import annotations

import numpy as np

from . import eclipses, frames, planets, positions, sidereal_time, timescales


def _horizon(spk, body, tt, site, jd_ut):
    """(azimuth from the south point going west, true altitude, apparent altitude) of a body seen from a site."""
    w, f = timescales.split_jd(float(tt))
    ra, dec, _ = positions.observe(spk, body, float(w), float(f), observer="site", site=site, jd_ut=jd_ut, equatorial=True)
    lst = float(sidereal_time.local_apparent_sidereal_time(jd_ut, site[0], tt))
    ha = lst - float(ra)
    lat = np.radians(site[1])
    d = float(dec)
    alt = np.arcsin(np.sin(lat) * np.sin(d) + np.cos(lat) * np.cos(d) * np.cos(ha))
    az = np.arctan2(np.sin(ha), np.cos(ha) * np.sin(lat) - np.tan(d) * np.cos(lat))
    true_alt = float(np.degrees(alt))
    return float(np.degrees(az) % 360.0), true_alt, true_alt + _refraction(true_alt)


def _refraction(h):
    if h < -1.0:
        return 0.0
    return 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11))) / 60.0 * (1013.25 / 1010.0) * (283.0 / 288.0)


def lunar_attributes(spk, tt, site, jd_ut):
    """Umbral and penumbral magnitude, the Moon's azimuth and altitudes, and its angular distance from opposition."""
    d, ru, rp = eclipses._lunar_metrics(spk, tt)
    r = eclipses.R_MOON
    um = max(0.0, (ru + r - d) / (2 * r))
    pm = max(0.0, (rp + r - d) / (2 * r))
    az, alt, app = _horizon(spk, planets.MOON, tt, site, jd_ut)
    rs, rm = eclipses._sun_moon(spk, tt)
    cosang = float(-(rs / np.linalg.norm(rs)) @ (rm / np.linalg.norm(rm)))
    opposition = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    return {"umbral": um, "penumbral": pm, "azimuth": az, "altitude": alt, "apparent_altitude": app,
            "opposition_distance": opposition}


def _overlap_fraction(rs, rm, sep):
    """Fraction of the solar disc covered by the lunar disc (angular radii and centre separation, same units)."""
    if sep >= rs + rm:
        return 0.0
    if sep <= abs(rs - rm):
        return 1.0 if rm >= rs else (rm / rs) ** 2
    a = rs * rs * np.arccos((sep * sep + rs * rs - rm * rm) / (2 * sep * rs))
    b = rm * rm * np.arccos((sep * sep + rm * rm - rs * rs) / (2 * sep * rm))
    c = 0.5 * np.sqrt((-sep + rs + rm) * (sep + rs - rm) * (sep - rs + rm) * (sep + rs + rm))
    return float((a + b - c) / (np.pi * rs * rs))


def solar_attributes(spk, tt, site, jd_ut):
    """Magnitude (fraction of the Sun's diameter covered), diameter ratio, obscuration, and where the Sun stands."""
    sep, rs, rm = eclipses._local_sep(spk, tt, site, jd_ut)
    mag = max(0.0, (rs + rm - sep) / (2 * rs))
    az, alt, app = _horizon(spk, planets.SUN, tt, site, jd_ut)
    rs_v, rm_v = eclipses._sun_moon(spk, tt)
    axis = rm_v - rs_v
    length = float(np.linalg.norm(axis))
    axis = axis / length
    obs, _ = positions.site_vector_true_equator(site[0], site[1], site[2], jd_ut, tt)
    beyond_moon = float((np.asarray(obs) - rm_v) @ axis)                  # distance past the Moon along the shadow axis
    umbra_radius = eclipses.R_MOON_SOLAR - beyond_moon * (eclipses.R_SUN - eclipses.R_MOON_SOLAR) / length
    return {"core_shadow_km": 2.0 * abs(umbra_radius), "magnitude": mag, "diameter_ratio": rm / rs, "obscuration": _overlap_fraction(rs, rm, sep),
            "azimuth": az, "altitude": alt, "apparent_altitude": app, "separation": float(np.degrees(sep))}
