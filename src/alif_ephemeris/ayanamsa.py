"""Sidereal zodiacs: ayanamsas.

An ayanamsa is the angle between the tropical zero point (the moving vernal equinox) and the sidereal zero
point (fixed against the stars): sidereal longitude = tropical longitude - ayanamsa.

Fixed-point systems
    Defined by one fact: at the epoch `t0`, the sidereal zero point sits at ecliptic longitude `a0` of that
    epoch's mean equinox. We carry that point through precession (IAU 2006) to any date, exactly, by rotating
    its direction from the ecliptic of t0 to the ecliptic of the date.

Empirical systems
    Some programs define their value by a published formula or table that is not a pure fixed point. Those are
    stored as the fitted polynomial in Julian centuries from J2000 and say so in `source`.

Star-based systems (true Chitra and friends) need star positions and come with the fixed-star module.

Values are mean ayanamsas (no nutation). Add `nutation` for the true value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import frames, timescales

J2000 = 2451545.0
R2D = 180.0 / np.pi
D2R = np.pi / 180.0


@dataclass(frozen=True)
class Ayanamsa:
    name: str
    kind: str                    # "fixed point" or "polynomial"
    t0: float = J2000            # epoch (Julian Day, TT) for a fixed point
    a0: float = 0.0              # degrees at t0 for a fixed point
    poly: tuple = ()             # polynomial coefficients (degrees, per Julian century from J2000)
    source: str = ""
    star: object = None          # a star name, a Hipparcos number or an (ra, dec) pair in degrees (ICRS) for "star"
    target: float = 0.0          # sidereal longitude (degrees) the star or point is held at, for "star"


def _mean_equator_to_ecliptic(jd_tt):
    return frames.mean_obliquity(jd_tt)


def _precession_matrix(jd_tt):
    """GCRS -> mean equator and equinox of date (IAU 2006, frame bias included, no nutation)."""
    gamb, phib, psib, epsa = frames.fukushima_williams(jd_tt)
    return frames._rot_x(-epsa) @ frames._rot_z(-psib) @ frames._rot_x(phib) @ frames._rot_z(gamb)


def fixed_point_ayanamsa(a0_deg, t0_tt, jd_tt):
    """Ayanamsa (degrees, mean) at `jd_tt` for a zero point at ecliptic longitude a0 at epoch t0."""
    jd_tt = np.asarray(jd_tt, dtype=float)
    a = a0_deg * D2R
    e0 = float(_mean_equator_to_ecliptic(t0_tt))
    # direction of the zero point in the ecliptic of t0, then in the mean equator of t0
    v_ecl = np.array([np.cos(a), np.sin(a), 0.0])
    c, s = np.cos(e0), np.sin(e0)
    v_eq0 = np.array([v_ecl[0], c * v_ecl[1], s * v_ecl[1]])
    gcrs = _precession_matrix(t0_tt).T @ v_eq0
    veq = np.einsum("...ij,j->...i", _precession_matrix(jd_tt), gcrs)
    e = _mean_equator_to_ecliptic(jd_tt)
    y = np.cos(e) * veq[..., 1] + np.sin(e) * veq[..., 2]
    x = veq[..., 0]
    return np.mod(np.arctan2(y, x) * R2D + 180.0, 360.0) - 180.0


def star_ayanamsa(star, target_deg, jd_tt):
    """Ayanamsa that holds a star (or a fixed point on the sky) at sidereal longitude `target_deg`.

    The star's mean longitude of date (proper motion and precession, no nutation or aberration) minus the target.
    A star is a name or Hipparcos number; a point is an (ra, dec) pair in degrees that does not move.
    """
    from . import stars
    jd_tt = np.asarray(jd_tt, dtype=float)
    if isinstance(star, tuple):
        ra, dec = np.radians(star[0]), np.radians(star[1])
        d = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
        gamb, phib, psib, epsa = frames.fukushima_williams(jd_tt)
        pm = frames._rot_x(-epsa) @ frames._rot_z(-psib) @ frames._rot_x(phib) @ frames._rot_z(gamb)
        eq = np.einsum("...ij,j->...i", pm, d)
        lon, _, _ = frames.to_spherical(frames.equator_to_ecliptic(eq, frames.mean_obliquity(jd_tt)))
        lon = np.degrees(lon)
    else:
        lon, _ = stars.mean_ecliptic(stars.find(star), jd_tt)
    return np.mod(lon - target_deg + 180.0, 360.0) - 180.0


# The galactic pole of the IAU 1958 definition (B1950 right ascension 192.25, declination 27.4), turned into a J2000
# direction with the standard B1950 to J2000 rotation.
_B1950_TO_J2000 = np.array([[0.9999256794956877, -0.0111814832204662, -0.0048590038153592],
                            [0.0111814832391717, 0.9999374848933135, -0.0000271625947142],
                            [0.0048590037723143, -0.0000271702937440, 0.9999881946023742]])


def _galactic_pole_j2000():
    ra, dec = np.radians(192.25), np.radians(27.4)
    return _B1950_TO_J2000 @ np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def galactic_node_longitude(jd_tt):
    """Tropical longitude (mean equinox of date) where the galactic equator crosses the ecliptic going north."""
    jd_tt = np.asarray(jd_tt, dtype=float)
    pole = _galactic_pole_j2000()
    p = np.einsum("...ij,j->...i", _precession_matrix(jd_tt), pole)
    e = _mean_equator_to_ecliptic(jd_tt)
    ye = np.cos(e) * p[..., 1] + np.sin(e) * p[..., 2]
    # the node direction is (0, 0, 1) x pole_ecliptic = (-ye, x, 0)
    return np.mod(np.degrees(np.arctan2(p[..., 0], -ye)), 360.0)


def value(ay: Ayanamsa, jd_tt):
    """Mean ayanamsa in degrees at Terrestrial Time `jd_tt`."""
    jd_tt = np.asarray(jd_tt, dtype=float)
    if ay.kind == "galactic node":                       # a0 is added to (node longitude - 240 degrees)
        return np.mod(galactic_node_longitude(jd_tt) - 240.0 + ay.a0 + 180.0, 360.0) - 180.0
    if ay.kind == "fixed point":
        return fixed_point_ayanamsa(ay.a0, ay.t0, jd_tt)
    if ay.kind == "star":
        return star_ayanamsa(ay.star, ay.target, jd_tt)
    t = (jd_tt - J2000) / 36525.0
    return np.polyval(ay.poly[::-1], t)


def true_value(ay: Ayanamsa, jd_tt):
    """True ayanamsa: the mean value plus nutation in longitude."""
    dpsi, _ = frames.nutation_iau2000a(np.asarray(jd_tt, dtype=float))
    return value(ay, jd_tt) + np.asarray(dpsi) * R2D


# Systems whose definition is a fixed point are filled in below; see docs/AYANAMSAS.md for the sources.
SYSTEMS: dict[str, Ayanamsa] = {}


def register(ay: Ayanamsa):
    SYSTEMS[ay.name] = ay
    return ay


# Star based systems. The star (or point) is held at a fixed sidereal longitude.
#   Spica at 180 degrees is what the program JHora prints as "Lahiri": measured against 518 of its charts from
#   1835 to 2010 the two agree to better than 0.02 arcsec (tools/check_star_ayanamsa.py).
_GALACTIC_CENTRE = (266.416837, -29.007810)       # Sgr A*, ICRS (Reid and Brunthaler 2004)
register(Ayanamsa("Lahiri (JHora)", "star", star="Spica", target=180.0,
                  source="Spica at 180 degrees; equals the value JHora prints as Lahiri"))
register(Ayanamsa("True Chitra (Spica 180)", "star", star="Spica", target=180.0, source="Spica at 180 degrees"))
register(Ayanamsa("True Revati (Zeta Piscium 359-50)", "star", star="Revati", target=359.0 + 50.0 / 60.0,
                  source="zeta Piscium at 359 deg 50 min"))
register(Ayanamsa("True Pushya (Delta Cancri 106)", "star", star="Asellus Australis", target=106.0,
                  source="delta Cancri at 106 degrees"))
register(Ayanamsa("True Mula (Lambda Scorpii 240)", "star", star="Shaula", target=240.0,
                  source="lambda Scorpii at 240 degrees"))
register(Ayanamsa("Galactic Centre 0 Sagittarius", "star", star=_GALACTIC_CENTRE, target=240.0,
                  source="the galactic centre at 0 deg Sagittarius"))
register(Ayanamsa("Galactic Centre (Gil Brand)", "star", star=_GALACTIC_CENTRE, target=244.376941,
                  source="the galactic centre at 4 deg 22 min 37 sec Sagittarius"))
register(Ayanamsa("True Sheoran (Delta Cancri)", "star", star="Asellus Australis", target=103.492642,
                  source="delta Cancri held 2.507358 degrees before 106 degrees"))
register(Ayanamsa("Galactic equator (IAU 1958)", "galactic node", a0=0.0,
                  source="the node of the IAU 1958 galactic equator on the ecliptic of date, held at 240 degrees"))
register(Ayanamsa("Galactic equator (true)", "galactic node", a0=0.05292,
                  source="the IAU 1958 galactic equator node, with the true-equator shift of 0.05292 degrees"))
register(Ayanamsa("Galactic equator (Mula)", "galactic node", a0=-6.613747,
                  source="the IAU 1958 galactic equator node moved to the Mula nakshatra start"))
register(Ayanamsa("Galactic Centre 0 Capricorn (Cochrane)", "star", star=_GALACTIC_CENTRE, target=270.0,
                  source="the galactic centre at 0 deg Capricorn"))


# Fixed-point systems: the mean ayanamsa at J2000.0 (degrees) is the numerical definition of each convention.
# They are carried to other dates by exact precession (fixed_point_ayanamsa). Checked in docs/AYANAMSAS.md.
_FIXED_AT_J2000 = [
("Fagan-Bradley", 24.740299966),
    ("Lahiri", 23.857092325),
    ("DeLuce", 27.815752774),
    ("Raman", 22.410791012),
    ("Usha-Shashi", 20.057541000),
    ("Krishnamurti (KP)", 23.760240012),
    ("Djwhal Khul", 28.359678600),
    ("Yukteshwar", 22.478803000),
    ("JN Bhasin", 22.762137000),
    ("Babylonian (Kugler 1)", 23.533639863),
    ("Babylonian (Kugler 2)", 24.933639863),
    ("Babylonian (Kugler 3)", 25.783639863),
    ("Babylonian (Huber)", 24.733639863),
    ("Babylonian (Eta Piscium)", 24.522527935),
    ("Aldebaran 15 Taurus (fixed)", 24.758923883),
    ("Hipparchos", 20.247788094),
    ("Sassanian", 19.992959383),
    ("J2000", 0.000000000),
    ("J1900", 1.396581000),
    ("B1950", 0.698370129),
    ("Suryasiddhanta", 20.895058836),
    ("Suryasiddhanta (mean Sun)", 20.680424886),
    ("Aryabhata", 20.895059722),
    ("Aryabhata (mean Sun)", 20.657427342),
    ("Suryasiddhanta (Revati)", 20.103388376),
    ("Suryasiddhanta (Citra)", 23.005763276),
    ("Galactic (Mardyks)", 30.017793877),
    ("Aryabhata 522", 20.575846967),
    ("Babylonian (Britton)", 24.615752774),
    ("Galactic equator (Fiorenza)", 25.000019094),
    ("Valens (Moon)", 22.795609299),
    ("Lahiri 1940", 23.842323232),
    ("Lahiri (VP285)", 23.863481202),
    ("Krishnamurti (VP291)", 23.780364957),
    ("Lahiri (ICRC)", 23.856788988),
]
for _name, _a0 in _FIXED_AT_J2000:
    register(Ayanamsa(_name, "fixed point", t0=J2000, a0=_a0,
                      source="J2000 value of the standard convention; see docs/AYANAMSAS.md"))


def names():
    """The names of every registered ayanamsa."""
    return list(SYSTEMS)


def get(name):
    """Look up an ayanamsa by name (case-insensitive, extra spaces ignored)."""
    if isinstance(name, Ayanamsa):
        return name
    key = " ".join(str(name).split()).lower()
    for k, v in SYSTEMS.items():
        if k.lower() == key:
            return v
    raise KeyError(f"unknown ayanamsa {name!r}; see ayanamsa.names()")


def ayanamsa(name, jd_tt, true: bool = False):
    """Ayanamsa in degrees for a registered system at Terrestrial Time `jd_tt` (mean, or true with nutation)."""
    ay = get(name)
    return true_value(ay, jd_tt) if true else value(ay, jd_tt)
