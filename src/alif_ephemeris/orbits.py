"""Osculating orbital elements, nodes and apsides, and distance extremes of the planets.

Elements are the two-body (Kepler) orbit that fits the body's heliocentric position and velocity at one moment, on the
mean ecliptic and equinox of date. Standard formulas; nothing is read from any other ephemeris program.
"""
from __future__ import annotations

import numpy as np

from . import frames, planets, timescales

GM_SUN = 132712440041.279419            # km^3/s^2 (DE440)
# Sun to planet system mass ratios (DE440 values)
MASS_RATIO = {planets.MERCURY: 6023657.945, planets.VENUS: 408523.721, planets.EARTH: 332946.0487,
              planets.MARS: 3098703.59, planets.JUPITER: 1047.348644, planets.SATURN: 3497.9018,
              planets.URANUS: 22902.9816, planets.NEPTUNE: 19412.2598, planets.PLUTO: 1.35e8}
AU = planets.AU_KM
DAY = 86400.0
EARTH_PERIOD_DAYS = 365.256363
PRECESSION_DEG_PER_YEAR = 50.2879 / 3600.0


def _barycentric_pid(body):
    return {planets.JUPITER: 5, planets.SATURN: 6, planets.URANUS: 7, planets.NEPTUNE: 8, planets.PLUTO: 9}.get(body, body)


def _date_ecliptic_matrix(jd_tt):
    """ICRS to the mean ecliptic and equinox of J2000 (the frame elements are quoted in)."""
    eps0 = 84381.406 * frames.ASEC2RAD
    c, s = np.cos(eps0), np.sin(eps0)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def heliocentric_state(spk, body, jd_tt):
    """Position (AU) and velocity (AU/day) relative to the Sun, mean ecliptic of date."""
    whole, frac = (float(x) for x in timescales.split_jd(float(jd_tt)))
    pid = _barycentric_pid(body)
    pb, vb = planets._barycentric(spk, pid, whole, frac)
    ps, vs = planets._barycentric(spk, planets.SUN, whole, frac)
    m = _date_ecliptic_matrix(jd_tt)
    return m @ ((np.asarray(pb) - np.asarray(ps)) / AU), m @ ((np.asarray(vb) - np.asarray(vs)) * DAY / AU)


def elements(spk, body, jd_tt):
    """Dictionary of osculating elements of a planet (Sun-centred)."""
    r, v = heliocentric_state(spk, body, jd_tt)
    ratio = MASS_RATIO.get(body, MASS_RATIO.get(_barycentric_pid(body), 1.0e12))
    mu = GM_SUN * (1.0 + 1.0 / ratio) * DAY * DAY / AU ** 3          # AU^3/day^2
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    evec = np.cross(v, h) / mu - r / rn
    e = float(np.linalg.norm(evec))
    a = 1.0 / (2.0 / rn - vn * vn / mu)
    inc = float(np.degrees(np.arccos(h[2] / hn)))
    node_vec = np.array([-h[1], h[0], 0.0])
    nn = np.linalg.norm(node_vec)
    omega_node = float(np.degrees(np.arctan2(node_vec[1], node_vec[0])) % 360.0)
    arg_peri = float(np.degrees(np.arccos(np.clip(node_vec @ evec / (nn * e), -1, 1))))
    if evec[2] < 0:
        arg_peri = 360.0 - arg_peri
    nu = float(np.degrees(np.arccos(np.clip(evec @ r / (e * rn), -1, 1))))
    if r @ v < 0:
        nu = 360.0 - nu
    nu_r = np.radians(nu)
    ecc_anom = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu_r / 2), np.sqrt(1 + e) * np.cos(nu_r / 2))
    mean_anom = float(np.degrees(ecc_anom - e * np.sin(ecc_anom)) % 360.0)
    ecc_anom = float(np.degrees(ecc_anom) % 360.0)
    n = np.sqrt(mu / a ** 3)                                          # rad/day
    period_days = 2.0 * np.pi * np.sqrt(a ** 3 / (GM_SUN * DAY * DAY / AU ** 3))     # Kepler's third law with the Sun's mass
    n = 2.0 * np.pi / period_days
    peri_long = (omega_node + arg_peri) % 360.0
    sidereal_years = period_days / 365.25
    deg_per_year = 360.0 / sidereal_years
    return {"a": float(a), "e": e, "incl": inc, "node": omega_node, "arg_peri": arg_peri, "peri_long": peri_long,
            "mean_anomaly": mean_anom, "true_anomaly": nu, "ecc_anomaly": ecc_anom,
            "mean_longitude": (peri_long + mean_anom) % 360.0, "sidereal_period_years": sidereal_years,
            "mean_daily_motion": float(np.degrees(n)), "tropical_period_years": 360.0 / (deg_per_year + PRECESSION_DEG_PER_YEAR),
            "synodic_days": float(1.0 / (1.0 / EARTH_PERIOD_DAYS - 1.0 / period_days)),
            "perihelion_jd": float(jd_tt - np.radians(mean_anom) / n),
            "perihelion_dist": float(a * (1 - e)), "aphelion_dist": float(a * (1 + e))}


def element_tuple(el):
    """The elements in the order of the reference function's result (first 17 slots)."""
    return (el["a"], el["e"], el["incl"], el["node"], el["arg_peri"], el["peri_long"], el["mean_anomaly"],
            el["true_anomaly"], el["ecc_anomaly"], el["mean_longitude"], el["sidereal_period_years"],
            el["mean_daily_motion"], el["tropical_period_years"], el["synodic_days"], el["perihelion_jd"],
            el["perihelion_dist"], el["aphelion_dist"])


def point_on_orbit(el, true_anomaly_deg):
    """Heliocentric ecliptic (J2000) position of the orbit point at a given true anomaly (AU)."""
    nu = np.radians(true_anomaly_deg)
    r = el["a"] * (1 - el["e"] ** 2) / (1 + el["e"] * np.cos(nu))
    u = np.radians(el["arg_peri"]) + nu
    om, i = np.radians(el["node"]), np.radians(el["incl"])
    return r * np.array([np.cos(om) * np.cos(u) - np.sin(om) * np.sin(u) * np.cos(i),
                         np.sin(om) * np.cos(u) + np.cos(om) * np.sin(u) * np.cos(i),
                         np.sin(u) * np.sin(i)])


def distance_extremes(spk, body, jd_tt):
    """(largest possible, smallest possible, current) distance from the Earth, from the two orbits' size alone."""
    el = elements(spk, body, jd_tt)
    earth = elements(spk, planets.EARTH, jd_tt)
    mx = el["aphelion_dist"] + earth["aphelion_dist"]
    if el["a"] > earth["a"]:
        mn = el["perihelion_dist"] - earth["aphelion_dist"]
    else:
        mn = earth["perihelion_dist"] - el["aphelion_dist"]
    rp, _ = heliocentric_state(spk, body, jd_tt)
    re, _ = heliocentric_state(spk, planets.EARTH, jd_tt)
    return float(mx), float(abs(mn)), float(np.linalg.norm(rp - re))


def j2000_ecliptic_to_date(vec, jd_tt):
    """Rotate a vector from the J2000 ecliptic to the true ecliptic and equinox of date."""
    from . import frames
    m = _date_ecliptic_matrix(jd_tt)                         # ICRS to J2000 ecliptic
    icrs = m.T @ np.asarray(vec, dtype=float)
    equ = frames.bias_precession_nutation(jd_tt) @ icrs
    return frames.equator_to_ecliptic(equ, frames.true_obliquity(jd_tt))
