"""The public interface: ask for a body at a date, get its apparent place.

Example
-------
>>> from alif_ephemeris import Ephemeris, SUN, julian_day
>>> eph = Ephemeris("data/raw/de440s.bsp")
>>> pos = eph.apparent(SUN, jd_ut=julian_day(2000, 1, 1, 12))
>>> round(pos.longitude, 3)
280.369

Longitudes and latitudes are in degrees on the true ecliptic and equinox of date (tropical).
Distances are in AU. Speeds are per day.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import ayanamsa as _ayanamsa, frames, lunar_points, planets, stars as _stars, timescales
from .spk import SPK, SPKSet

# Body names -> NAIF ids used by this library
SUN = planets.SUN
MOON = planets.MOON
MERCURY = planets.MERCURY
VENUS = planets.VENUS
MARS = planets.MARS
JUPITER = planets.JUPITER
SATURN = planets.SATURN
URANUS = planets.URANUS
NEPTUNE = planets.NEPTUNE
PLUTO = planets.PLUTO

DEG = 180.0 / np.pi
D2R = np.pi / 180.0


@dataclass(frozen=True)
class Position:
    longitude: float          # degrees, 0..360
    latitude: float           # degrees
    distance: float           # AU
    longitude_speed: float    # degrees per day
    latitude_speed: float     # degrees per day
    distance_speed: float     # AU per day
    jd_tt: float              # the Terrestrial Time used, for the record
    delta_t_error_arcsec: float = 0.0   # how far the longitude could move if Delta-T is off by one sigma
    ayanamsa: float = 0.0               # degrees subtracted to get a sidereal longitude (0 when tropical)


class Ephemeris:
    """An open ephemeris file. Safe to share between threads (it holds no mutable state)."""

    def __init__(self, path):
        """`path` is one .bsp file, a list of them, or several joined with the OS path separator."""
        if isinstance(path, str):
            paths = [p for p in path.split(os.pathsep) if p] if os.pathsep in path and not os.path.exists(path) else [path]
        else:
            paths = list(path)
        self.spk = SPK(paths[0]) if len(paths) == 1 else SPKSet(paths)

    # -- time ----------------------------------------------------------------------------
    @staticmethod
    def tt_from_ut(jd_ut):
        return timescales.ut_to_tt(jd_ut)

    # -- positions -----------------------------------------------------------------------
    def apparent_tt(self, body: int, whole_tt, frac_tt=0.0, **kw):
        """Apparent (lon, lat, distance) in radians/radians/AU for Terrestrial Time whole_tt + frac_tt."""
        return planets.apparent_ecliptic(self.spk, body, whole_tt, frac_tt, **kw)

    def apparent(self, body: int, jd_ut: float | None = None, jd_tt: float | None = None,
                 speed: bool = True, sidereal: str | None = None) -> Position:
        """Apparent geocentric place of a body. Give either `jd_ut` or `jd_tt`.

        Pass `sidereal="Lahiri"` (any name in `ayanamsa.names()`) for a sidereal longitude: the apparent
        longitude minus the true ayanamsa (mean plus nutation).

        `delta_t_error_arcsec` is the effect of Delta-T uncertainty on the longitude (zero when you
        give Terrestrial Time directly). It is usually the largest error in a real chart.
        """
        if (jd_ut is None) == (jd_tt is None):
            raise ValueError("give exactly one of jd_ut or jd_tt")
        if jd_tt is not None:
            whole, frac = timescales.split_jd(jd_tt)
        else:
            whole, frac = timescales.ut_to_tt_parts(jd_ut)
        whole, frac = float(whole), float(frac)
        tt = whole + frac
        if not speed:
            lon, lat, dist = (float(x) for x in self.apparent_tt(body, whole, frac))
            ay = float(_ayanamsa.ayanamsa(sidereal, tt, true=True)) if sidereal else 0.0
            return Position((lon * DEG - ay) % 360.0, lat * DEG, dist, 0.0, 0.0, 0.0, tt, 0.0, ay)

        # One batched pass at five nearby times; speed by Richardson extrapolation of central
        # differences (error well below 1e-9 degrees a day).
        steps = np.array([0.0, 0.005, -0.005, 0.01, -0.01])
        lon, lat, dist = self.apparent_tt(body, whole, frac + steps)
        lon, lat, dist = np.asarray(lon), np.asarray(lat), np.asarray(dist)
        ay = np.zeros(5)
        if sidereal:
            ay = np.asarray(_ayanamsa.ayanamsa(sidereal, tt + steps, true=True))
            lon = lon - ay * D2R

        def diff(i_plus, i_minus, h):
            dl = (lon[i_plus] - lon[i_minus] + np.pi) % (2 * np.pi) - np.pi
            return np.array([dl, lat[i_plus] - lat[i_minus], dist[i_plus] - dist[i_minus]]) / (2 * h)

        d = (4.0 * diff(1, 2, 0.005) - diff(3, 4, 0.01)) / 3.0
        lon_speed = float(d[0] * DEG)
        err = 0.0
        if jd_ut is not None:
            err = abs(lon_speed) * 3600.0 * timescales.delta_t_uncertainty(jd_ut) / 86400.0
        return Position(float(lon[0] * DEG % 360.0), float(lat[0] * DEG), float(dist[0]), lon_speed,
                        float(d[1] * DEG), float(d[2]), tt, err, float(ay[0]))

    def coverage(self):
        """First and last Julian Day (TDB) covered by the ephemeris file, as (start, end)."""
        return self.spk.coverage()

    def apparent_many(self, body: int, jd_ut, sidereal: str | None = None):
        """Vectorised: apparent longitude and latitude (degrees) and distance (AU) for many dates."""
        whole, frac = timescales.ut_to_tt_parts(np.asarray(jd_ut, dtype=float))
        lon, lat, dist = self.apparent_tt(body, whole, frac)
        lon = lon * DEG
        if sidereal:
            lon = (lon - _ayanamsa.ayanamsa(sidereal, whole + frac, true=True)) % 360.0
        return lon, lat * DEG, dist

    # -- lunar points and fixed stars ---------------------------------------------------------
    LUNAR_POINTS = ("mean node", "true node", "mean apogee", "osculating apogee")

    def lunar_point(self, kind: str, jd_ut: float, sidereal: str | None = None) -> float:
        """Longitude (degrees, true equinox of date) of a lunar point.

        kind: "mean node", "true node", "mean apogee" (mean Black Moon Lilith), "osculating apogee".
        The south node and the perigee are 180 degrees from the node and apogee.
        """
        whole, frac = timescales.ut_to_tt_parts(jd_ut)
        whole, frac = float(whole), float(frac)
        tt = whole + frac
        dpsi = float(np.degrees(frames.nutation_iau2000a(tt)[0]))
        if kind == "mean node":
            lon = float(lunar_points.mean_node(tt)) + dpsi
        elif kind == "mean apogee":
            lon = float(lunar_points.mean_apogee(tt)) + dpsi
        elif kind == "true node":
            lon = float(lunar_points.true_node(self.spk, whole, frac))
        elif kind == "osculating apogee":
            lon = float(lunar_points.osculating_apogee(self.spk, whole, frac)[0])
        else:
            raise ValueError(f"kind must be one of {self.LUNAR_POINTS}")
        if sidereal:
            lon -= float(_ayanamsa.ayanamsa(sidereal, tt, true=True))
        return lon % 360.0

    def fixed_star(self, name, jd_ut: float, sidereal: str | None = None):
        """Apparent (longitude, latitude) in degrees of a fixed star (IAU name or Hipparcos number)."""
        whole, frac = timescales.ut_to_tt_parts(jd_ut)
        whole, frac = float(whole), float(frac)
        lon, lat = _stars.apparent(self.spk, _stars.find(name), whole, frac)
        lon = float(lon)
        if sidereal:
            lon -= float(_ayanamsa.ayanamsa(sidereal, whole + frac, true=True))
        return lon % 360.0, float(lat)
