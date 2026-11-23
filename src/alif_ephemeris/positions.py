"""General positions: choose the observer, the corrections and the reference frame.

`observe()` is the flexible form of the pipeline in planets.py. It can place the observer at the Earth's
centre, on the Earth's surface, at the Sun's centre or at the solar-system barycentre; switch light time,
light bending and aberration on or off; give coordinates of date (with or without nutation) or J2000; and
return ecliptic or equatorial angles.

All angles are returned in radians, distances in AU.
"""
from __future__ import annotations

import numpy as np

from . import frames, planets, sidereal_time, timescales

AU_KM = planets.AU_KM
EARTH_RADIUS_KM = 6378.137
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_OMEGA = 7.2921150e-5                # rad/s


def site_vector_true_equator(lon_east_deg, lat_deg, height_m, jd_ut, jd_tt):
    """Observer's position (km) and velocity (km/s) in the true equator and equinox of date."""
    lat = np.radians(lat_deg)
    e2 = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)
    n = EARTH_RADIUS_KM / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    h = height_m / 1000.0
    xy = (n + h) * np.cos(lat)
    z = (n * (1.0 - e2) + h) * np.sin(lat)
    theta = sidereal_time.local_apparent_sidereal_time(jd_ut, lon_east_deg, jd_tt)
    pos = np.stack([xy * np.cos(theta), xy * np.sin(theta), z * np.ones_like(theta)], axis=-1)
    vel = np.stack([-EARTH_OMEGA * xy * np.sin(theta), EARTH_OMEGA * xy * np.cos(theta), np.zeros_like(theta)], axis=-1)
    return pos, vel


def observe(spk, body, whole_tt, frac_tt=0.0, *, observer="earth", site=None, jd_ut=None,
            light_time=True, deflection=True, aberration=True, nutation=True, frame="date",
            equatorial=False, cartesian=False):
    """Position of `body` as seen from the chosen observer.

    observer: "earth" (geocentre), "sun" (heliocentric), "ssb" (barycentric) or "site" with `site` =
        (east longitude deg, latitude deg, height m) and `jd_ut` for the Earth's rotation.
    frame: "date" for the true (or, with nutation=False, mean) equator/ecliptic of date; "j2000" for the
        mean J2000 equator and ecliptic.
    Returns (lon_or_ra, lat_or_dec, distance) in radians and AU, or an (x, y, z) vector in AU if `cartesian`.
    """
    whole = np.asarray(whole_tt, dtype=float)
    frac_tt = np.asarray(frac_tt, dtype=float)
    jd_tt = whole + frac_tt
    frac = frac_tt + timescales.tdb_minus_tt(jd_tt) / planets.SECONDS_PER_DAY
    sec = planets.SECONDS_PER_DAY

    # ---- observer state (barycentric, ICRS) ----
    earth_p, earth_v = planets._barycentric(spk, planets.EARTH, whole, frac)
    if observer == "earth":
        obs_p, obs_v = earth_p, earth_v
    elif observer == "sun":
        obs_p, obs_v = planets._barycentric(spk, planets.SUN, whole, frac)
    elif observer == "ssb":
        obs_p, obs_v = np.zeros_like(earth_p), np.zeros_like(earth_v)
    elif observer == "site":
        lon, lat, height = site
        ut = jd_ut if jd_ut is not None else jd_tt - np.asarray(timescales.delta_t(jd_tt)) / sec
        r_eq, v_eq = site_vector_true_equator(lon, lat, height, ut, jd_tt)
        m = frames.bias_precession_nutation(jd_tt)
        mt = np.swapaxes(m, -1, -2)
        obs_p = earth_p + np.einsum("...ij,...j->...i", mt, r_eq)
        obs_v = earth_v + np.einsum("...ij,...j->...i", mt, v_eq)
    else:
        raise ValueError("observer must be earth, sun, ssb or site")

    # ---- target, with light time ----
    tau = np.zeros_like(frac)
    vec = None
    for _ in range(4 if light_time else 1):
        p, _v = planets._barycentric(spk, body, whole, frac - tau)
        vec = p - obs_p
        if not light_time:
            break
        tau = np.linalg.norm(vec, axis=-1) / planets.C_KM_S / sec
    dist_km = np.linalg.norm(vec, axis=-1, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(dist_km > 0, vec / np.where(dist_km > 0, dist_km, 1.0), 0.0)
    geocentric_like = observer in ("earth", "site")
    p = unit
    if deflection and geocentric_like and np.all(dist_km > 0):
        pos_au = vec / AU_KM
        for dbody, rmass, planet in planets.DEFLECTORS:
            if planet == body or dbody == body:
                continue
            pos_au = pos_au + planets._deflection(spk, dbody, rmass, pos_au, obs_p / AU_KM, whole, frac)
        p = pos_au / np.linalg.norm(pos_au, axis=-1, keepdims=True)
    if aberration and geocentric_like and np.all(dist_km > 0):
        sun_p, _ = planets._barycentric(spk, planets.SUN, whole, frac)
        em = np.linalg.norm(obs_p - sun_p, axis=-1, keepdims=True) / AU_KM
        v = obs_v / planets.C_KM_S
        pdv = np.sum(p * v, axis=-1, keepdims=True)
        bm1 = np.sqrt(1.0 - np.sum(v * v, axis=-1, keepdims=True))
        w1 = 1.0 + pdv / (1.0 + bm1)
        p = p * bm1 + w1 * v + (planets.SRS / em) * (v - pdv * p)
        p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    vec_au = p * dist_km / AU_KM

    # ---- frame ----
    if frame == "j2000":
        gamb, phib, psib, epsa = frames.fukushima_williams(2451545.0)
        m = frames._rot_x(-epsa) @ frames._rot_z(-psib) @ frames._rot_x(phib) @ frames._rot_z(gamb)   # frame bias
        eq = np.einsum("ij,...j->...i", m, vec_au)
        obl = frames.mean_obliquity(2451545.0)
    else:
        if nutation:
            m = frames.bias_precession_nutation(jd_tt)
            obl = frames.true_obliquity(jd_tt)
        else:
            gamb, phib, psib, epsa = frames.fukushima_williams(jd_tt)
            m = frames._rot_x(-epsa) @ frames._rot_z(-psib) @ frames._rot_x(phib) @ frames._rot_z(gamb)
            obl = frames.mean_obliquity(jd_tt)
        eq = np.einsum("...ij,...j->...i", m, vec_au)
    out = eq if equatorial else frames.equator_to_ecliptic(eq, obl)
    if cartesian:
        return out
    return frames.to_spherical(out)
