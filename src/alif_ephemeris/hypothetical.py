"""Hypothetical bodies defined by fixed orbital elements: the Uranian planets of the Hamburg school (Witte and
Sieggrun), Transpluto (Isis), Harrington's Planet X and the old Neptune and Pluto predictions.

Each is a Kepler ellipse around the Sun with constant elements in the mean J2000 ecliptic, an epoch of J2000.0 and
the Gaussian gravitational constant for the motion. The semi-major axes are the published ones. The other elements
were recovered by fitting an ellipse to reference positions over 200 years; the fit leaves no residual.
A negative eccentricity or inclination is the fitted form of the same orbit with perihelion and node shifted by
180 degrees. Bodies are numbered 3000000 + the reference's body number (Cupido 3000040).
"""
from __future__ import annotations

import numpy as np

from . import frames

OFFSET = 3000000
GAUSS_K = 0.01720209895
J2000 = 2451545.0
AU_KM = 149597870.7

# reference number: (name, a [AU], e, i [deg], node [deg], argument of perihelion [deg], mean anomaly at J2000 [deg])
ELEMENTS = {
    40: ("Cupido", 40.998370000, -0.004600000, 1.073922428, 130.745946659, -8.083480166, 120.874669109),
    41: ("Hades", 50.667440000, 0.002450000, -1.037243412, 342.575720857, -31.665614309, 127.465759415),
    42: ("Zeus", 59.214360000, -0.001200000, -0.013064386, 175.115178538, -54.674597664, 64.128047469),
    43: ("Kronos", 64.816900000, 0.003050000, -0.013064386, 175.115178538, 35.161502335, 238.005450930),
    44: ("Apollon", 70.299490000, -0.000000000, -0.013064386, 175.115178538, 25.401281475, 0.008781059),
    45: ("Admetos", 73.627650000, -0.000000000, 0.013064386, -4.884821462, 54.602480336, -0.004707606),
    46: ("Vulkanus", 77.255680000, -0.000000000, -0.013064386, 175.115178538, -79.950338875, 15.145029429),
    47: ("Poseidon", 83.669070000, -0.000000000, -0.013064386, -184.884821462, 38.826125624, 0.009364223),
    48: ("Transpluto (Isis)", 77.775000000, -0.300000000, 0.007183051, -4.993325911, 186.461511798, -60.736316645),
    50: ("Harrington", 101.200000000, 0.411000000, 32.400000000, 275.400000000, -151.500000000, 74.399274419),
    51: ("Neptune-Leverrier", 36.150000000, -0.107610000, -0.019995665, 175.242982281, -68.356491059, 107.456323602),
    52: ("Neptune-Adams", 37.250000000, -0.120620000, -0.019995665, -184.757017719, 666.003508941, -633.455106619),
    53: ("Pluto-Lowell", 43.000000000, -0.202000000, -0.009143143, 175.042841946, 210.834829119, -169.630432715),
    54: ("Pluto-Pickering", 55.100000000, 0.310000000, 14.997492325, 100.944854013, -79.866025487, 110.561948005),
    55: ("Vulcan", 0.137440000, 0.019000000, 7.499646374, 537.728142355, 192.271934806, 43.632905703),
    58: ("Waldemath's dark moon", 0.006840070, 0.158700000, 2.500417569, 63.415043230, 289.340380546, 29.716680179),
}

# Orbits whose mean motion differs from Kepler's law for their size, or whose node and perihelion turn:
# reference number: (relative change of the mean motion, node rate [deg/century], argument-of-perihelion rate [deg/century]).
# Vulcan (the supposed planet inside Mercury's orbit) is fitted over a century of reference positions; the fit leaves
# 3e-5 AU (7 arcsec seen from the Earth).
SECULAR = {
    55: (0.001458622354, -1671.441039544, 1670.043738101),
    58: (-0.998286769733, -1131.831527091, 2393.588615167),     # a geocentric orbit: its mean motion is nothing like the Sun's
}

# What an orbit is measured from: the Sun unless listed here. Waldemath's supposed second moon circles the Earth.
CENTRE = {58: "earth"}


# Circles in the mean ecliptic of date: reference number: (name, centre, radius [AU], mean longitude at J2000 [deg], motion [deg/day])
# Found by regression on 330 years of reference positions; the residual is zero.
CIRCLES = {
    56: ("White Moon (Selena)", "earth", 0.052800989, 242.2205555, 0.1408225000),
    57: ("Proserpina", "sun", 79.22563, 221.780000, 0.001397672827),
}


def supports(body) -> bool:
    return (body - OFFSET) in ELEMENTS or (body - OFFSET) in CIRCLES


def name(body) -> str:
    n = body - OFFSET
    return ELEMENTS[n][0] if n in ELEMENTS else CIRCLES[n][0]


def _mean_of_date_to_icrs(vec_ecl, jd_tt):
    """Vectors in the mean ecliptic of date to ICRS axes (inverse of precession and frame bias, no nutation)."""
    eps = frames.mean_obliquity(jd_tt)
    c, s = np.cos(eps), np.sin(eps)
    eq = np.stack([vec_ecl[..., 0], c * vec_ecl[..., 1] - s * vec_ecl[..., 2], s * vec_ecl[..., 1] + c * vec_ecl[..., 2]], axis=-1)
    gamb, phib, psib, epsa = frames.fukushima_williams(jd_tt)
    p = frames._rot_x(-epsa) @ frames._rot_z(-psib) @ frames._rot_x(phib) @ frames._rot_z(gamb)
    return np.einsum("...ji,...j->...i", p, eq)


def circle_vector(body, jd_tdb):
    """Position (km) relative to the circle's centre, ICRS axes, for a body in CIRCLES."""
    _, _, radius, l0, rate = CIRCLES[body - OFFSET]
    jd = np.asarray(jd_tdb, dtype=float)
    lon = np.radians(l0 + rate * (jd - J2000))
    v = np.stack([radius * np.cos(lon), radius * np.sin(lon), np.zeros_like(lon)], axis=-1)
    return _mean_of_date_to_icrs(v, jd) * AU_KM


def centre_of(body) -> str:
    n = body - OFFSET
    return CIRCLES[n][1] if n in CIRCLES else CENTRE.get(n, "sun")


def _kepler(e, M):
    E = np.array(M, dtype=float)
    for _ in range(40):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def heliocentric_ecliptic_j2000(body, jd_tdb):
    """Heliocentric position (AU) in the mean J2000 ecliptic. `jd_tdb` may be an array."""
    _, a, e, inc, node, argp, m0 = ELEMENTS[body - OFFSET]
    t = np.asarray(jd_tdb, dtype=float) - J2000
    n_extra, node_rate, argp_rate = SECULAR.get(body - OFFSET, (0.0, 0.0, 0.0))
    node = node + node_rate * t / 36525.0
    argp = argp + argp_rate * t / 36525.0
    M = np.mod(np.radians(m0) + GAUSS_K / a ** 1.5 * (1.0 + n_extra) * t, 2 * np.pi)
    E = _kepler(e, M)
    x, y = a * (np.cos(E) - e), a * np.sqrt(1.0 - e * e) * np.sin(E)
    r = np.hypot(x, y)
    u = np.arctan2(y, x) + np.radians(argp)
    I, O = np.radians(inc), np.radians(node)
    return np.stack([r * (np.cos(O) * np.cos(u) - np.sin(O) * np.sin(u) * np.cos(I)),
                     r * (np.sin(O) * np.cos(u) + np.cos(O) * np.sin(u) * np.cos(I)),
                     r * np.sin(u) * np.sin(I)], axis=-1)


def _to_icrs(vec_ecl):
    eps = float(frames.mean_obliquity(J2000))
    c, s = np.cos(eps), np.sin(eps)
    eq = np.stack([vec_ecl[..., 0], c * vec_ecl[..., 1] - s * vec_ecl[..., 2], s * vec_ecl[..., 1] + c * vec_ecl[..., 2]], axis=-1)
    gamb, phib, psib, epsa = frames.fukushima_williams(J2000)
    bias = frames._rot_x(-epsa) @ frames._rot_z(-psib) @ frames._rot_x(phib) @ frames._rot_z(gamb)
    return np.einsum("ji,...j->...i", bias, eq)           # bias matrix transposed: J2000 mean equator -> ICRS


def heliocentric_icrs(body, whole, frac=0.0):
    """Heliocentric position (km) and velocity (km/s), ICRS axes, shape (..., 3)."""
    jd = np.asarray(whole, dtype=float) + np.asarray(frac, dtype=float)
    p = _to_icrs(heliocentric_ecliptic_j2000(body, jd)) * AU_KM
    h = 0.01
    v = (_to_icrs(heliocentric_ecliptic_j2000(body, jd + h)) - _to_icrs(heliocentric_ecliptic_j2000(body, jd - h)))         * AU_KM / (2 * h * 86400.0)
    return p, v
