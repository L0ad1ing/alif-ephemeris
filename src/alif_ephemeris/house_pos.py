"""In which house is a point? Returns a number from 1.0 up to (but not including) 13.0: the house number plus how far
through the house the point is.

Each system measures "how far" its own way, from the same geometry that defines its cusps:
- Placidus: the point's own diurnal and nocturnal semi-arcs.
- Alcabitius: the semi-arcs of the Ascendant's declination (so western houses mirror the eastern ones).
- Koch: oblique ascension and descension, measured against the Midheaven point.
- Regiomontanus: the great circle through the north and south points of the horizon, read where it meets the equator.
- Campanus: the same circle, read on the prime vertical.
- Axial rotation (Meridian) and Morinus: equal slices of the equator, read by right ascension (Morinus through the
  point's ecliptic longitude).
- Systems defined by their cusps (Equal, Porphyry, Whole Sign, Vehlow, Sripati, ...): linear between the two
  surrounding cusps of the point's ecliptic longitude.
All formulas follow the published geometry and were checked against the reference program to 1e-4 of a house.
"""
from __future__ import annotations

import numpy as np

from . import houses as _houses


def _wrap(x):
    return (x + 180.0) % 360.0 - 180.0


def _radec(eps, lon, lat):
    l, b = np.radians(lon), np.radians(lat)
    ra = np.arctan2(np.sin(l) * np.cos(eps) - np.tan(b) * np.sin(eps), np.cos(l))
    dec = np.arcsin(np.sin(b) * np.cos(eps) + np.cos(b) * np.sin(eps) * np.sin(l))
    return float(np.degrees(ra) % 360.0), float(np.degrees(dec))


def _asc(armc, lat, eps):
    a, phi = np.radians(armc), np.radians(lat)
    return float(np.degrees(np.arctan2(np.cos(a), -(np.sin(a) * np.cos(eps) + np.tan(phi) * np.sin(eps)))) % 360.0)


def _wrap_house(x):
    return 1.0 + ((x - 1.0) % 12.0)


def _semi_arc_position(h, dsa):
    """Position from hour angle h (west positive) and semi-arcs dsa (day) and 180 - dsa (night)."""
    nsa = 180.0 - dsa
    if -dsa <= h <= 0.0:
        return _wrap_house(10.0 - 3.0 * h / dsa)
    if 0.0 < h <= dsa:
        return 10.0 - 3.0 * h / dsa
    if h < -dsa:
        return 1.0 + 3.0 * (-h - dsa) / nsa
    return 4.0 + 3.0 * (180.0 - h) / nsa


def placidus(armc, lat, eps, lon, blat):
    ra, dec = _radec(eps, lon, blat)
    h = _wrap(armc - ra)
    x = -np.tan(np.radians(lat)) * np.tan(np.radians(dec))
    if abs(x) >= 1.0:
        return 0.0
    return _semi_arc_position(h, float(np.degrees(np.arccos(x))))


def alcabitius(armc, lat, eps, lon, blat):
    a = np.radians(_asc(armc, lat, eps))
    dec_a = np.arcsin(np.sin(eps) * np.sin(a))
    x = -np.tan(np.radians(lat)) * np.tan(dec_a)
    if abs(x) >= 1.0:
        return 0.0
    dsa = float(np.degrees(np.arccos(x)))
    nsa = 180.0 - dsa
    ra, _ = _radec(eps, lon, blat)
    h = _wrap(armc - ra)
    if -dsa <= h <= 0.0:
        return _wrap_house(10.0 - 3.0 * h / dsa)
    if 0.0 < h <= nsa:
        return 10.0 - 3.0 * h / nsa
    if h > nsa:
        return 7.0 - 3.0 * (h - nsa) / dsa
    return 1.0 + 3.0 * (-h - dsa) / nsa


def _mc_longitude(armc, eps):
    a = np.radians(armc)
    return float(np.degrees(np.arctan2(np.sin(a), np.cos(a) * np.cos(eps))) % 360.0)


def koch(armc, lat, eps, lon, blat):
    ra, dec = _radec(eps, lon, blat)
    phi = np.radians(lat)
    x = np.tan(phi) * np.tan(np.radians(dec))
    if abs(x) >= 1.0:
        return 0.0
    ad = float(np.degrees(np.arcsin(x)))
    h = _wrap(armc - ra)
    dmc = np.arcsin(np.sin(eps) * np.sin(np.radians(_mc_longitude(armc, eps))))
    y = np.tan(phi) * np.tan(dmc)
    if abs(y) >= 1.0:
        return 0.0
    admc = float(np.degrees(np.arcsin(y)))
    dsa = 90.0 + ad
    oa = ra - ad
    if -dsa <= h <= 0.0:
        return 10.0 + 3.0 * _wrap(oa - (armc - admc)) / (90.0 + admc)
    if h < -dsa:
        return 1.0 + 3.0 * _wrap(oa - (armc + 90.0)) / (90.0 + admc)
    if 0.0 < h <= dsa:
        s = ra + ad + 90.0
        return 10.0 - 3.0 * _wrap(armc + admc + 90.0 - s) / (90.0 + admc)
    q = koch(armc, lat, eps, (lon + 180.0) % 360.0, -blat)          # west below the horizon: the opposite point
    return 0.0 if q == 0.0 else _wrap_house(q + 6.0)


def _horizon_vector(armc, lat, eps, lon, blat):
    ra, dec = _radec(eps, lon, blat)
    h, phi, d = np.radians(_wrap(armc - ra)), np.radians(lat), np.radians(dec)
    east = -np.cos(d) * np.sin(h)
    up = np.sin(d) * np.sin(phi) + np.cos(d) * np.cos(h) * np.cos(phi)
    return east, up


def campanus(armc, lat, eps, lon, blat):
    east, up = _horizon_vector(armc, lat, eps, lon, blat)
    theta = float(np.degrees(np.arctan2(up, east)))
    return 1.0 + ((-theta / 30.0) % 12.0)


def regiomontanus(armc, lat, eps, lon, blat):
    east, up = _horizon_vector(armc, lat, eps, lon, blat)
    theta = np.arctan2(up, east)
    hv = float(np.degrees(np.arctan2(-np.cos(np.radians(lat)) * np.cos(theta), np.sin(theta))))
    return _wrap_house(10.0 - hv / 30.0)


def axial(armc, lat, eps, lon, blat):
    ra, _ = _radec(eps, lon, blat)
    return _wrap_house(10.0 - _wrap(armc - ra) / 30.0)


def morinus(armc, lat, eps, lon, blat):
    l = np.radians(lon)
    alpha = float(np.degrees(np.arctan2(np.sin(l), np.cos(l) * np.cos(eps))) % 360.0)
    return _wrap_house(10.0 - _wrap(armc - alpha) / 30.0)


def from_cusps(cusps, lon):
    """Linear between the two cusps around the longitude."""
    for k in range(12):
        a, b = cusps[k], cusps[(k + 1) % 12]
        span = (b - a) % 360.0
        if span == 0.0:
            span = 360.0
        if (lon - a) % 360.0 < span:
            return k + 1 + ((lon - a) % 360.0) / span
    return 1.0


_SPECIAL = {"Placidus": placidus, "Alcabitius": alcabitius, "Koch": koch, "Regiomontanus": regiomontanus,
            "Campanus": campanus, "Meridian": axial, "Morinus": morinus}


def house_position(armc, lat, eps_deg, lon, blat, system="Placidus"):
    """House position (1.0 to 12.999) of a point at ecliptic longitude `lon` and latitude `blat` (degrees)."""
    eps = np.radians(eps_deg)
    fn = _SPECIAL.get(system)
    if fn is not None:
        return float(fn(armc % 360.0, lat, eps, lon % 360.0, blat))
    cusps = _houses.compute(armc, eps_deg, lat, system).cusps
    return float(from_cusps(cusps, lon % 360.0))
