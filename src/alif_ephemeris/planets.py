"""Apparent geocentric positions of the Sun, Moon and planets.

Pipeline (the standard one used by astronomical almanacs, IAU conventions):
  1. Barycentric positions from the JPL ephemeris.
  2. Light-time: where the body was when the light left it.
  3. Gravitational light bending by the Sun, Jupiter and Saturn.
  4. Annual aberration (relativistic form).
  5. Bias, precession and nutation to the true equator and equinox of date.
  6. Rotate to the true ecliptic of date and read off longitude and latitude.

Written from the published formulas (IERS Conventions 2010, the Explanatory Supplement to
the Astronomical Almanac). Pure functions: no global state, safe to call from many threads.

Times are passed as two parts, (whole, fraction): whole days (a half-integer Julian Day) and the
rest. One float Julian Day can only resolve about 40 microseconds, which is 1 to 2 metres of
Earth's motion, so the pipeline never adds a small number to a full Julian Day.
"""
from __future__ import annotations

import numpy as np

from . import centres, frames, hypothetical, smallbodies, timescales
from .spk import SPK

C_KM_S = 299792.458
AU_KM = 149597870.7
SRS = 1.97412574336e-8        # twice the Sun's gravitational radius, in AU (2GM/c^2)
SECONDS_PER_DAY = 86400.0
C_AU_DAY = C_KM_S * SECONDS_PER_DAY / AU_KM

planets_SUN = 10
# NAIF ids used here
SUN, MOON, EARTH = 10, 301, 399
MERCURY, VENUS, MARS = 199, 299, 4
JUPITER, SATURN, URANUS, NEPTUNE, PLUTO = 599, 699, 799, 899, 999   # planet centres (see centres.py)
# Mars: the planet file gives the Mars system barycentre. Mars's moons are so small that the centre
# is within about 0.2 metres of it, so no offset is applied.
# Jupiter to Pluto: barycentre plus the planet-centre offset table (centres.py).
# Raw barycentre ids (5 to 9) still work if you want the system barycentres.

# Bodies that bend light enough to matter: (id, Sun mass divided by its mass, the planet that is this body).
DEFLECTORS = ((SUN, 1.0, SUN), (5, 1047.3486, JUPITER), (6, 3497.898, SATURN))


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


_CACHE: dict = {}
_CACHE_MAX = 8192


def _barycentric(spk: SPK, body: int, whole, frac=0.0):
    """Barycentric state of a body, ICRS axes (km, km/s). Single times are remembered: one position computation
    asks for the Earth and Sun at the same instant several times."""
    if np.ndim(whole) == 0 and np.ndim(frac) == 0:
        key = (id(spk), body, float(whole), float(frac))
        hit = _CACHE.get(key)
        if hit is not None:
            return hit
        out = _barycentric_uncached(spk, body, whole, frac)
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        for a in out:
            a.flags.writeable = False
        _CACHE[key] = out
        return out
    return _barycentric_uncached(spk, body, whole, frac)


# ids for the plain DE440 barycentres of Uranus and Neptune (used by the Swiss-style layer, which follows the
# reference program's older orbit model)
PLAIN_DE_BARYCENTRES = {7007: 7, 7008: 8}


def _barycentric_uncached(spk: SPK, body: int, whole, frac=0.0):
    if body >= hypothetical.OFFSET:                   # bodies defined by orbital elements or a circle
        jd = np.asarray(whole, dtype=float) + np.asarray(frac, dtype=float)
        n = body - hypothetical.OFFSET
        if n in hypothetical.CIRCLES:
            centre = hypothetical.centre_of(body)
            hp = hypothetical.circle_vector(body, jd)
            h = 0.01
            hv = (hypothetical.circle_vector(body, jd + h) - hypothetical.circle_vector(body, jd - h)) / (2 * h * 86400.0)
            cp, cv = _barycentric(spk, EARTH if centre == "earth" else SUN, whole, frac)
            return hp + cp, hv + cv
        hp, hv = hypothetical.heliocentric_icrs(body, whole, frac)
        if hypothetical.centre_of(body) == "earth":                # an orbit around the Earth rather than the Sun
            cp, cv = _barycentric(spk, EARTH, whole, frac)
            return hp + cp, hv + cv
        sp, sv = spk.barycentric(planets_SUN, whole, frac)
        return hp + np.moveaxis(np.asarray(sp), 0, -1), hv + np.moveaxis(np.asarray(sv), 0, -1)
    if body >= smallbodies.OFFSET:                    # asteroids and centaurs
        p, v = smallbodies.barycentric(body, whole, frac)
        return np.moveaxis(p, 0, -1), np.moveaxis(v, 0, -1)
    if body in PLAIN_DE_BARYCENTRES:                  # raw DE440 barycentre, without the newer Uranus/Neptune orbit fit
        p, v = spk.barycentric(PLAIN_DE_BARYCENTRES[body], whole, frac)
        return np.moveaxis(np.asarray(p), 0, -1), np.moveaxis(np.asarray(v), 0, -1)
    base = centres.BARYCENTRE_OF.get(body, body)
    p, v = spk.barycentric(base, whole, frac)
    p, v = np.asarray(p), np.asarray(v)
    if base in centres.CORRECTED_BARYCENTRES:        # newer Uranus and Neptune orbits
        cp, cv = centres.offset(1000 + base, whole, frac)
        p, v = p + cp, v + cv
    if body in centres.BARYCENTRE_OF:                # planet centre rather than system barycentre
        op, ov = centres.offset(body, whole, frac)
        p, v = p + op, v + ov
    return np.moveaxis(p, 0, -1), np.moveaxis(v, 0, -1)


def _light_time_direction(spk: SPK, body: int, whole, frac, earth_p):
    """Astrometric vector Earth -> body (km) with light travel time removed."""
    tau = np.zeros_like(np.asarray(frac, dtype=float))
    for _ in range(4):
        p, _v = _barycentric(spk, body, whole, frac - tau)
        vec = p - earth_p
        tau = np.linalg.norm(vec, axis=-1) / C_KM_S / SECONDS_PER_DAY
    return vec


def _deflection(spk, dbody, rmass, position, observer, whole, frac):
    """Change in the apparent vector (AU) caused by one gravitating body.

    Explanatory Supplement to the Astronomical Almanac (3rd ed.), light deflection. The bending
    body is placed where it was when the light passed closest to it.
    """
    tlt = np.linalg.norm(position, axis=-1, keepdims=True) / C_AU_DAY
    bpos, _ = _barycentric(spk, dbody, whole, frac)
    gpv = bpos / AU_KM - observer
    phat = _unit(position)
    dlt = np.sum(phat * gpv, axis=-1, keepdims=True) / C_AU_DAY
    back = np.clip(dlt, 0.0, tlt)[..., 0]
    bclose, _ = _barycentric(spk, dbody, whole, frac - back)
    pe = observer - bclose / AU_KM
    pq = position + pe
    pmag = np.linalg.norm(position, axis=-1, keepdims=True)
    qhat = _unit(pq)
    emag = np.linalg.norm(pe, axis=-1, keepdims=True)
    ehat = pe / emag
    pdotq = np.sum(phat * qhat, axis=-1, keepdims=True)
    qdote = np.sum(qhat * ehat, axis=-1, keepdims=True)
    edotp = np.sum(ehat * phat, axis=-1, keepdims=True)
    on_line = np.abs(edotp) > 0.99999999999
    fac1 = SRS / (emag * rmass)
    d = fac1 * (pdotq * ehat - edotp * qhat) / (1.0 + qdote) * pmag
    return np.where(on_line, 0.0, d)


def astrometric_icrf(spk: SPK, body: int, whole_tt, frac_tt=0.0):
    """Astrometric geocentric vector (km) in the ICRF: light time only, no deflection or aberration.

    This is the quantity catalogues and ephemeris services call the astrometric place. It does not
    depend on any precession or nutation model, so it is the cleanest way to test the ephemeris itself.
    """
    whole = np.asarray(whole_tt, dtype=float)
    frac_tt = np.asarray(frac_tt, dtype=float)
    frac = frac_tt + timescales.tdb_minus_tt(whole + frac_tt) / SECONDS_PER_DAY
    earth_p, _ = _barycentric(spk, EARTH, whole, frac)
    return _light_time_direction(spk, body, whole, frac, earth_p)


def apparent_vector_true_equator(spk: SPK, body: int, whole_tt, frac_tt=0.0, deflection: bool = True,
                                 aberration: bool = True):
    """Apparent geocentric vector (AU) of `body` in the true equator and equinox of date.

    Time is Terrestrial Time given as `whole_tt + frac_tt` days (see module notes).
    """
    whole = np.asarray(whole_tt, dtype=float)
    frac_tt = np.asarray(frac_tt, dtype=float)
    jd_tt = whole + frac_tt                                    # for slowly varying terms only
    frac = frac_tt + timescales.tdb_minus_tt(jd_tt) / SECONDS_PER_DAY   # TDB, for the ephemeris

    earth_p, earth_v = _barycentric(spk, EARTH, whole, frac)
    sun_p, _ = _barycentric(spk, SUN, whole, frac)

    vec_km = _light_time_direction(spk, body, whole, frac, earth_p)
    dist_km = np.linalg.norm(vec_km, axis=-1, keepdims=True)
    p = vec_km / dist_km

    sun_to_obs = earth_p - sun_p
    em = np.linalg.norm(sun_to_obs, axis=-1, keepdims=True) / AU_KM

    if deflection:
        pos_au = vec_km / AU_KM
        for dbody, rmass, planet in DEFLECTORS:
            if planet == body or dbody == body:
                continue
            pos_au = pos_au + _deflection(spk, dbody, rmass, pos_au, earth_p / AU_KM, whole, frac)
        p = _unit(pos_au)

    if aberration:
        v = earth_v / C_KM_S                    # velocity in units of c
        pdv = np.sum(p * v, axis=-1, keepdims=True)
        bm1 = np.sqrt(1.0 - np.sum(v * v, axis=-1, keepdims=True))
        w1 = 1.0 + pdv / (1.0 + bm1)
        w2 = SRS / em
        p = p * bm1 + w1 * v + w2 * (v - pdv * p)
        p = _unit(p)

    vec_gcrs = p * dist_km / AU_KM
    m = frames.bias_precession_nutation(jd_tt)
    return np.einsum("...ij,...j->...i", m, vec_gcrs)


def apparent_ecliptic(spk: SPK, body: int, whole_tt, frac_tt=0.0, **kw):
    """Apparent geocentric (longitude, latitude, distance) on the true ecliptic of date.

    Longitude and latitude in radians, distance in AU.
    """
    v = apparent_vector_true_equator(spk, body, whole_tt, frac_tt, **kw)
    jd_tt = np.asarray(whole_tt, dtype=float) + np.asarray(frac_tt, dtype=float)
    ecl = frames.equator_to_ecliptic(v, frames.true_obliquity(jd_tt))
    return frames.to_spherical(ecl)
