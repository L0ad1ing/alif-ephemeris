"""Fixed stars from the ESA Hipparcos catalogue.

9,827 stars brighter than magnitude 6.6, with the IAU's official proper names where they exist.
Positions and proper motions are at epoch J1991.25 (ICRS). A star's place at another date comes from its
proper motion and parallax; the apparent place then goes through the same light bending, aberration,
precession and nutation as the planets.

Radial velocity is not in the catalogue and is ignored. Over a few centuries that changes a star's place by
well under a milli-arcsecond for every naked-eye star except a handful of very fast ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import frames, planets, timescales

MAS = np.pi / 180.0 / 3600.0 / 1000.0
EPOCH_JD = 2448349.0625          # J1991.25 (TT)
AU_PER_PARSEC = 206264.806247
J2000 = 2451545.0


@lru_cache(maxsize=1)
def _catalogue():
    d = np.load(Path(__file__).with_name("data") / "stars.npz")
    cat = {k: d[k] for k in d}
    cat["_by_hip"] = {int(h): i for i, h in enumerate(cat["hip"])}
    cat["_by_name"] = {}
    for i, n in enumerate(cat["name"]):
        if n:
            cat["_by_name"][str(n).lower()] = i
    return cat


@dataclass(frozen=True)
class Star:
    hip: int
    name: str
    designation: str
    constellation: str
    magnitude: float
    index: int


@lru_cache(maxsize=1)
def _bayer():
    """Bayer code (alLeo, al-2Lib, beGem ...) to Hipparcos number, for the IAU-named stars."""
    path = Path(__file__).with_name("data") / "star_bayer.txt"
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    return out


def find(key) -> Star:
    """Look up a star by IAU name (case-insensitive) or Hipparcos number."""
    cat = _catalogue()
    if isinstance(key, (int, np.integer)) or (isinstance(key, str) and key.strip().isdigit()):
        i = cat["_by_hip"].get(int(key))
    else:
        i = cat["_by_name"].get(str(key).strip().lower())
        if i is None:                                           # Bayer codes as astrology software writes them: alLeo, al-2Lib
            hip = _bayer().get(str(key).strip().lstrip(","))
            i = cat["_by_hip"].get(hip) if hip else None
    if i is None:
        raise KeyError(f"star {key!r} not found (Hipparcos stars brighter than V 6.6 and IAU names)")
    return Star(int(cat["hip"][i]), str(cat["name"][i]), str(cat["designation"][i]),
                str(cat["constellation"][i]), float(cat["vmag"][i]), int(i))


def named_stars():
    """All stars that have an official IAU name."""
    cat = _catalogue()
    return [find(int(cat["hip"][i])) for i in np.flatnonzero(cat["name"] != "")]


def _unit_and_motion(i):
    cat = _catalogue()
    ra, dec = np.radians(cat["ra"][i]), np.radians(cat["dec"][i])
    u = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    north = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)])
    pm = (east * cat["pmra"][i] + north * cat["pmdec"][i]) * MAS          # radians per Julian year (pmra includes cos dec)
    plx = max(float(cat["plx"][i]), 0.1) * MAS                              # radians
    return u, pm, plx


def barycentric_direction(star: Star, jd_tt):
    """Unit vector (ICRS) from the solar-system barycentre to the star at date `jd_tt`, from proper motion."""
    u, pm, _ = _unit_and_motion(star.index)
    years = (np.asarray(jd_tt, dtype=float) - EPOCH_JD) / 365.25
    v = u + np.multiply.outer(years, pm)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def mean_ecliptic(star: Star, jd_tt):
    """Mean ecliptic longitude and latitude (degrees) of date: proper motion and precession only."""
    jd_tt = np.asarray(jd_tt, dtype=float)
    d = barycentric_direction(star, jd_tt)
    gamb, phib, psib, epsa = frames.fukushima_williams(jd_tt)
    p = frames._rot_x(-epsa) @ frames._rot_z(-psib) @ frames._rot_x(phib) @ frames._rot_z(gamb)
    eq = np.einsum("...ij,...j->...i", p, d)
    ecl = frames.equator_to_ecliptic(eq, frames.mean_obliquity(jd_tt))
    lon, lat, _ = frames.to_spherical(ecl)
    return np.degrees(lon), np.degrees(lat)


def apparent(spk, star: Star, whole_tt, frac_tt=0.0, deflection: bool = True, aberration: bool = True,
             nutation: bool = True, frame: str = "date", equatorial: bool = False):
    """Apparent (longitude, latitude) or (RA, Dec) in degrees.

    Default: the true ecliptic and equinox of date. nutation=False gives the mean equinox of date,
    frame="j2000" the mean J2000 frame, equatorial=True right ascension and declination.
    """
    whole = np.asarray(whole_tt, dtype=float)
    frac_tt = np.asarray(frac_tt, dtype=float)
    jd_tt = whole + frac_tt
    frac = frac_tt + timescales.tdb_minus_tt(jd_tt) / planets.SECONDS_PER_DAY
    _, _, plx = _unit_and_motion(star.index)
    dist_au = 1.0 / plx
    earth_p, earth_v = planets._barycentric(spk, planets.EARTH, whole, frac)
    star_bary = barycentric_direction(star, jd_tt) * dist_au
    vec_au = star_bary - earth_p / planets.AU_KM                 # geocentric, with parallax
    p = vec_au / np.linalg.norm(vec_au, axis=-1, keepdims=True)
    if deflection:
        pos = p * 1.0e3                                           # a long, finite distance in AU (direction is what matters)
        for dbody, rmass, _planet in planets.DEFLECTORS:
            pos = pos + planets._deflection(spk, dbody, rmass, pos, earth_p / planets.AU_KM, whole, frac)
        p = pos / np.linalg.norm(pos, axis=-1, keepdims=True)
    if aberration:
        sun_p, _ = planets._barycentric(spk, planets.SUN, whole, frac)
        em = np.linalg.norm(earth_p - sun_p, axis=-1, keepdims=True) / planets.AU_KM
        v = earth_v / planets.C_KM_S
        pdv = np.sum(p * v, axis=-1, keepdims=True)
        bm1 = np.sqrt(1.0 - np.sum(v * v, axis=-1, keepdims=True))
        w1 = 1.0 + pdv / (1.0 + bm1)
        p = p * bm1 + w1 * v + (planets.SRS / em) * (v - pdv * p)
        p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    if frame == "j2000":
        gamb, phib, psib, epsa = frames.fukushima_williams(2451545.0)
        m = frames._rot_x(-epsa) @ frames._rot_z(-psib) @ frames._rot_x(phib) @ frames._rot_z(gamb)
        eq = np.einsum("ij,...j->...i", m, p)
        obl = frames.mean_obliquity(2451545.0)
    elif nutation:
        eq = np.einsum("...ij,...j->...i", frames.bias_precession_nutation(jd_tt), p)
        obl = frames.true_obliquity(jd_tt)
    else:
        gamb, phib, psib, epsa = frames.fukushima_williams(jd_tt)
        m = frames._rot_x(-epsa) @ frames._rot_z(-psib) @ frames._rot_x(phib) @ frames._rot_z(gamb)
        eq = np.einsum("...ij,...j->...i", m, p)
        obl = frames.mean_obliquity(jd_tt)
    out = eq if equatorial else frames.equator_to_ecliptic(eq, obl)
    lon, lat, _ = frames.to_spherical(out)
    return np.degrees(lon), np.degrees(lat)
