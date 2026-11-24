"""Chart angles and house systems.

Everything here is derived from the published geometric definitions of each system (the great circles
that define the houses, intersected with the ecliptic), using standard spherical astronomy
(Meeus, Astronomical Algorithms; Holden; Munkasey). Pure functions, angles in degrees unless noted.

Inputs: the local apparent sidereal time as right ascension of the Midheaven (RAMC), the true obliquity of
the ecliptic, and the geographic latitude. `houses()` takes a Julian Day, longitude and latitude and does the
sidereal time and obliquity for you.

In the local frame used internally: x points to the meridian on the equator (hour angle 0), y to the west
(hour angle +90), z to the north celestial pole. A point with hour angle H and declination d is
(cos d cos H, cos d sin H, sin d). The right ascension of a point is RAMC - H.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import frames, sidereal_time, timescales

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def _norm360(x):
    return np.mod(x, 360.0)


# -- angles -------------------------------------------------------------------------------

def ascendant_from(ramc_deg, eps_deg, lat_deg):
    """Ecliptic longitude of the point rising on the eastern horizon.

    The horizon meets the ecliptic in two opposite points. The Ascendant is the eastern one: it lies
    within 180 degrees after the Midheaven. Inside the polar circles the plain formula can return the
    western point, so it is checked and flipped.
    """
    if np.ndim(ramc_deg) == 0 and np.ndim(eps_deg) == 0 and np.ndim(lat_deg) == 0:      # one place and time: plain math
        import math
        th, e, p = math.radians(ramc_deg), math.radians(eps_deg), math.radians(lat_deg)
        asc = math.degrees(math.atan2(math.cos(th), -(math.sin(th) * math.cos(e) + math.tan(p) * math.sin(e)))) % 360.0
        mc = math.degrees(math.atan2(math.sin(th), math.cos(th) * math.cos(e))) % 360.0
        return asc if (asc - mc) % 360.0 < 180.0 else (asc + 180.0) % 360.0
    th, e, p = ramc_deg * D2R, eps_deg * D2R, lat_deg * D2R
    asc = _norm360(np.arctan2(np.cos(th), -(np.sin(th) * np.cos(e) + np.tan(p) * np.sin(e))) * R2D)
    mc = midheaven_from(ramc_deg, eps_deg)
    return np.where(np.mod(asc - mc, 360.0) < 180.0, asc, _norm360(asc + 180.0)) if np.ndim(asc) else (
        float(asc) if (asc - mc) % 360.0 < 180.0 else float((asc + 180.0) % 360.0))


def midheaven_from(ramc_deg, eps_deg):
    """Ecliptic longitude of the Midheaven (the ecliptic point on the meridian, south of the zenith)."""
    if np.ndim(ramc_deg) == 0 and np.ndim(eps_deg) == 0:
        import math
        th, e = math.radians(ramc_deg), math.radians(eps_deg)
        return math.degrees(math.atan2(math.sin(th), math.cos(th) * math.cos(e))) % 360.0
    th, e = ramc_deg * D2R, eps_deg * D2R
    return _norm360(np.arctan2(np.sin(th), np.cos(th) * np.cos(e)) * R2D)


def vertex_from(ramc_deg, eps_deg, lat_deg):
    """The western intersection of the prime vertical with the ecliptic.

    The prime vertical is the horizon of an observer at co-latitude (90 - latitude) whose meridian lies
    180 degrees away in sidereal time, so the vertex is that observer's Ascendant."""
    return ascendant_from(ramc_deg + 180.0, eps_deg, 90.0 - lat_deg)


def _ecl_to_vec_local(lon_deg, eps_deg, ramc_deg):
    """Unit vector in the local frame for the ecliptic point (lon, 0)."""
    lam, e, th = lon_deg * D2R, eps_deg * D2R, ramc_deg * D2R
    xi = np.cos(lam)
    yi = np.sin(lam) * np.cos(e)
    zi = np.sin(lam) * np.sin(e)
    return np.array([xi * np.cos(th) + yi * np.sin(th), xi * np.sin(th) - yi * np.cos(th), zi])


def _local_to_ecl_lon(v, eps_deg, ramc_deg):
    th, e = ramc_deg * D2R, eps_deg * D2R
    xi = v[0] * np.cos(th) + v[1] * np.sin(th)
    yi = v[0] * np.sin(th) - v[1] * np.cos(th)
    zi = v[2]
    x = xi
    y = yi * np.cos(e) + zi * np.sin(e)
    return _norm360(np.arctan2(y, x) * R2D)


def _in_arc(x, start, direction, length):
    """Position of longitude x along an arc that starts at `start` and runs in `direction` (+1 or -1)."""
    t = ((x - start) * direction) % 360.0
    return t if t <= length + 1e-9 else None


def _pick(cands, arc):
    """Choose among candidate longitudes the one on the arc (start, direction, length)."""
    start, direction, length = arc
    inside = [c for c in cands if _in_arc(c, start, direction, length) is not None]
    if inside:
        return float(inside[0])
    mid = start + direction * length / 2.0
    return float(min(cands, key=lambda c: abs((c - mid + 180.0) % 360.0 - 180.0)))


def _circle_ecliptic_cusp(normal, eps_deg, ramc_deg, arc):
    """Intersection of the great circle with the given normal (local frame) and the ecliptic.

    The circle meets the ecliptic in two opposite points; `arc` = (start, direction, length) says which
    stretch of the ecliptic the cusp belongs to.
    """
    th, e = ramc_deg * D2R, eps_deg * D2R
    pole_i = np.array([0.0, -np.sin(e), np.cos(e)])           # ecliptic pole in the local frame
    pole = np.array([pole_i[0] * np.cos(th) + pole_i[1] * np.sin(th),
                     pole_i[0] * np.sin(th) - pole_i[1] * np.cos(th), pole_i[2]])
    d = np.cross(normal, pole)
    n = np.linalg.norm(d)
    if n < 1e-14:
        return float("nan")
    a = _local_to_ecl_lon(d / n, eps_deg, ramc_deg)
    return _pick([a, _norm360(a + 180.0)], arc)


# -- systems based on circles through the north and south points of the horizon -------------------

def _horizon_ns(lat_deg):
    p = lat_deg * D2R
    return np.array([-np.sin(p), 0.0, np.cos(p)])           # the north point of the horizon


def _regiomontanus(ramc, eps, lat, arcs, asc, mc):
    n_pt = _horizon_ns(lat)
    cusps = {}
    for house, a in ((11, 30.0), (12, 60.0), (2, 120.0), (3, 150.0)):
        pt = np.array([np.cos(a * D2R), -np.sin(a * D2R), 0.0])     # equator point at hour angle -a
        cusps[house] = _circle_ecliptic_cusp(np.cross(n_pt, pt), eps, ramc, arcs[house])
    return cusps


def _campanus(ramc, eps, lat, arcs, asc, mc):
    p = lat * D2R
    n_pt = _horizon_ns(lat)
    east = np.array([0.0, -1.0, 0.0])
    zenith = np.array([np.cos(p), 0.0, np.sin(p)])
    cusps = {}
    for house, beta in ((11, 60.0), (12, 30.0), (2, -30.0), (3, -60.0)):
        b = beta * D2R
        pt = np.cos(b) * east + np.sin(b) * zenith              # point on the prime vertical
        cusps[house] = _circle_ecliptic_cusp(np.cross(n_pt, pt), eps, ramc, arcs[house])
    return cusps


# -- systems that project equator divisions onto the ecliptic -------------------------------------

def _lon_of_ra(ra_deg, eps_deg):
    """Ecliptic longitude of the ecliptic point with the given right ascension (hour circles)."""
    a, e = ra_deg * D2R, eps_deg * D2R
    return _norm360(np.arctan2(np.sin(a), np.cos(a) * np.cos(e)) * R2D)


def _dec_of_lon(lon_deg, eps_deg):
    return np.arcsin(np.sin(eps_deg * D2R) * np.sin(lon_deg * D2R)) * R2D


def _meridian(ramc, eps, lat, arcs, asc, mc):
    """Axial rotation: all twelve cusps are the ecliptic points with right ascension RAMC + 30 (h - 10)."""
    return {h: _lon_of_ra(ramc + 30.0 * (h - 10), eps) for h in range(1, 13)}


def _morinus(ramc, eps, lat, arcs, asc, mc):
    """Equal 30 degree steps of right ascension from RAMC, carried to the ecliptic along circles through
    the poles of the ecliptic. Cusps 1, 4, 7 and 10 are not the Ascendant and Midheaven."""
    e = eps * D2R
    out = {}
    for h in range(1, 13):
        a = (ramc + 30.0 * (h - 10)) * D2R
        out[h] = _norm360(np.arctan2(np.sin(a) * np.cos(e), np.cos(a)) * R2D)
    return out


def _semi_arc(dec_deg, lat_deg):
    x = -np.tan(lat_deg * D2R) * np.tan(dec_deg * D2R)
    if abs(x) >= 1.0:
        return None
    return np.arccos(x) * R2D                                # diurnal semi-arc in degrees


def _placidus(ramc, eps, lat, arcs, asc, mc):
    import math
    e = math.radians(eps)
    sin_e, cos_e = math.sin(e), math.cos(e)
    tan_phi = math.tan(math.radians(lat))
    atan2, sin, cos, asin, acos, tan = math.atan2, math.sin, math.cos, math.asin, math.acos, math.tan
    rad, deg = math.radians, math.degrees
    out = {}
    for h, frac, from_noon in ((11, 1.0 / 3, True), (12, 2.0 / 3, True), (2, 2.0 / 3, False), (3, 1.0 / 3, False)):
        lon = {11: mc + 30.0, 12: mc + 60.0, 2: asc + 30.0, 3: asc + 60.0}[h]  # starting guess
        base = ramc if from_noon else ramc + 180.0
        sgn = 1.0 if from_noon else -1.0
        scale = frac if from_noon else frac

        def g(x):
            t = -tan_phi * tan(asin(sin_e * sin(rad(x))))        # -tan(dec) tan(lat)
            if abs(t) >= 1.0:
                return None
            dsa = deg(acos(t))                                   # diurnal semi-arc
            ra = base + frac * dsa if from_noon else base - frac * (180.0 - dsa)
            a = rad(ra)
            return deg(atan2(sin(a), cos(a) * cos_e)) % 360.0

        def wrap(d):
            return (d + 180.0) % 360.0 - 180.0

        for _ in range(40):
            x1 = g(lon)
            if x1 is None:
                return None
            d1 = wrap(x1 - lon)
            if abs(d1) < 1e-8:
                lon = x1
                break
            x2 = g(x1)
            if x2 is None:
                return None
            d2 = wrap(x2 - x1)
            den = d2 - d1
            if abs(den) > 1e-14:
                lon = (lon - d1 * d1 / den) % 360.0           # Steffensen (Aitken) step
            else:
                lon = x2
        out[h] = float(lon)
    return out


def _koch(ramc, eps, lat, arcs, asc, mc):
    dec_mc = _dec_of_lon(mc, eps)
    dsa = _semi_arc(dec_mc, lat)
    if dsa is None:
        return None
    return {11: ascendant_from(ramc - 2.0 * dsa / 3.0, eps, lat), 12: ascendant_from(ramc - dsa / 3.0, eps, lat),
            2: ascendant_from(ramc + dsa / 3.0, eps, lat), 3: ascendant_from(ramc + 2.0 * dsa / 3.0, eps, lat)}


def _alcabitius(ramc, eps, lat, arcs, asc, mc):
    dec_a = _dec_of_lon(asc, eps)
    dsa = _semi_arc(dec_a, lat)
    if dsa is None:
        return None
    nsa = 180.0 - dsa
    return {11: _lon_of_ra(ramc + dsa / 3.0, eps), 12: _lon_of_ra(ramc + 2.0 * dsa / 3.0, eps),
            2: _lon_of_ra(ramc + 180.0 - 2.0 * nsa / 3.0, eps), 3: _lon_of_ra(ramc + 180.0 - nsa / 3.0, eps)}


def _topocentric(ramc, eps, lat, arcs, asc, mc):
    """Polich-Page: the Ascendant formula at sidereal time RAMC + a - 90 degrees, with the pole height
    scaled by 1/3 (houses 11 and 3) or 2/3 (houses 12 and 2)."""
    t = np.tan(lat * D2R)
    spec = ((11, 30.0, 1.0 / 3), (12, 60.0, 2.0 / 3), (2, 120.0, 2.0 / 3), (3, 150.0, 1.0 / 3))
    out = {}
    for h, a, f in spec:
        v = float(ascendant_from(ramc + a - 90.0, eps, np.arctan(t * f) * R2D))
        out[h] = _pick([v, _norm360(v + 180.0)], arcs[h])
    return out


def _carter(ramc, eps, lat, arcs, asc, mc):
    """Poli-equatorial: equal 30 degree steps of right ascension from the right ascension of the Ascendant."""
    e = eps * D2R
    ra_asc = np.degrees(np.arctan2(np.sin(asc * D2R) * np.cos(e), np.cos(asc * D2R)))
    return {h: float(_lon_of_ra(ra_asc + 30.0 * (h - 1), eps)) for h in range(1, 13)}


# -- Sunshine houses (Bob Makransky): the Sun's own daily circle sets the cusps -----------------------------------------------

def _sunshine(ramc, eps, lat, sun_dec):
    """Cusps 1 to 12 of Makransky's Sunshine houses, or None when the Sun does not rise or set at this place.

    The Sun's daily circle (the small circle through the Sun parallel to the equator) is cut into six equal parts of
    hour angle between sunrise and sunset, and six between sunset and sunrise. Each of the twelve points is carried onto
    the ecliptic along the great circle through it and the north and south points of the horizon. Sunrise gives the
    Ascendant, noon the Midheaven, sunset the Descendant and midnight the IC."""
    th, e, ph, dl = ramc * D2R, eps * D2R, lat * D2R, sun_dec * D2R
    c = -np.tan(ph) * np.tan(dl)
    if abs(c) >= 1.0:
        return None
    h0 = float(np.arccos(c))                                   # the Sun's semi-diurnal arc
    north = np.array([-np.sin(ph), 0.0, np.cos(ph)])            # frame: x on the meridian at the equator, y east, z north
    pole = np.array([0.0, -np.sin(e), np.cos(e)])               # the ecliptic pole with x at the vernal equinox
    pole = np.array([pole[0] * np.cos(th) + pole[1] * np.sin(th), -pole[0] * np.sin(th) + pole[1] * np.cos(th), pole[2]])

    def cusp(ha):
        p = np.array([np.cos(dl) * np.cos(ha), -np.cos(dl) * np.sin(ha), np.sin(dl)])
        w = p - (p @ north) * north
        w /= np.linalg.norm(w)
        n = np.cross(north, p)
        n /= np.linalg.norm(n)
        d = np.cross(n, pole)
        d /= np.linalg.norm(d)
        x = d if d @ w > 0 else -d                              # the intersection on the half circle through the point
        xv = np.array([x[0] * np.cos(th) - x[1] * np.sin(th), x[0] * np.sin(th) + x[1] * np.cos(th), x[2]])
        return float(np.degrees(np.arctan2(xv[1] * np.cos(e) + xv[2] * np.sin(e), xv[0])) % 360.0)

    out = {}
    for j, house in enumerate((1, 12, 11, 10, 9, 8, 7)):                         # sunrise to sunset
        out[house] = cusp(-h0 + j * (2.0 * h0) / 6.0)
    for j, house in enumerate((7, 6, 5, 4, 3, 2, 1)):                            # sunset to sunrise
        if house not in (7, 1):
            out[house] = cusp(h0 + j * (2.0 * np.pi - 2.0 * h0) / 6.0)
    return out


# -- Pullen systems: each quadrant's three houses are narrow-wide-narrow (the two outer houses are equal) ---------

def _pullen_sd_outer(q):
    """Pullen sinusoidal delta (W. Pullen's published definition). In the small quadrant (size q) the middle house is
    x = (q - 30) / 2 and the outer houses x + n with n = (90 - q) / 4; the outer house is therefore q / 4 + 7.5. The same
    formula carries over to a large quadrant (its outer houses are the same function of its size)."""
    if q < 30.0:                                       # a quadrant this narrow cannot hold a middle house: it gets size 0
        return q / 2.0
    return q / 4.0 + 7.5


def _sinusoidal_ratio(q):
    """Ratio r >= 1 of Pullen's sinusoidal-ratio system for a small quadrant q (degrees, 0 < q <= 90). The small quadrant
    holds houses of sizes r x, x, r x (so x = q / (2 r + 1)); the large quadrant holds r^3 x, r^4 x, r^3 x and fills
    180 - q: q r^3 (r + 2) / (2 r + 1) = 180 - q. The left side grows with r, so bisection finds r."""
    if q >= 90.0:
        return 1.0
    target = 180.0 - q
    lo, hi = 1.0, 2.0
    while q * hi ** 3 * (hi + 2.0) / (2.0 * hi + 1.0) < target and hi < 1e12:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if q * mid ** 3 * (mid + 2.0) / (2.0 * mid + 1.0) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _pullen_sr_outer(q):
    """Outer house width of a quadrant of size q (0 < q < 180) in Pullen's sinusoidal-ratio system. A quadrant smaller
    than 90 degrees has outer houses r x and a middle house x; a larger one has outer houses r^3 x and a middle house
    r^4 x, where the ratio r comes from the complementary small quadrant (180 - q)."""
    if q <= 0.0:
        return 0.0
    if q <= 90.0:
        r = _sinusoidal_ratio(q)
        return r * q / (2.0 * r + 1.0)
    small = 180.0 - q
    if small <= 0.0:
        return q / 3.0
    r = _sinusoidal_ratio(small)
    return r ** 3 * small / (2.0 * r + 1.0)


def _pullen(outer_fn):
    def build(ramc, eps, lat, arcs, asc, mc):
        out = {}
        for (a, b), (start, direction, q) in (((11, 12), arcs[11]), ((2, 3), arcs[2])):
            w = outer_fn(q)
            out[a] = start + direction * w
            out[b] = start + direction * (q - w)
        return out
    return build



def _krusinski(ramc, eps, lat, arcs, asc, mc):
    """Krusinski-Pisa-Goelzer: the great circle through the Ascendant and the zenith is cut into twelve equal
    30 degree arcs starting at the Ascendant (going below the horizon); each division is carried to the ecliptic
    along its hour circle."""
    e, th, phi = eps * D2R, ramc * D2R, lat * D2R
    la = asc * D2R
    a = np.array([np.cos(la), np.sin(la) * np.cos(e), np.sin(la) * np.sin(e)])          # Ascendant, equatorial vector
    z = np.array([np.cos(phi) * np.cos(th), np.cos(phi) * np.sin(th), np.sin(phi)])      # zenith
    out = {}
    for k in range(12):
        step = 30.0 * k * D2R
        p = np.cos(step) * a - np.sin(step) * z
        out[k + 1] = float(_lon_of_ra(np.arctan2(p[1], p[0]) * R2D, eps))
    return out


def _horizon_houses(ramc, eps, lat, arcs, asc, mc):
    """Horizon (azimuthal) houses: the horizon is cut into twelve equal 30 degree arcs starting at the east point and
    going through the south towards the west in the northern hemisphere (the other way in the southern one). Each
    cusp is where the vertical circle through a division meets the ecliptic, on the half-circle of that azimuth."""
    e, th, phi = eps * D2R, ramc * D2R, lat * D2R
    zenith = np.array([np.cos(phi) * np.cos(th), np.cos(phi) * np.sin(th), np.sin(phi)])
    north = np.array([-np.sin(phi) * np.cos(th), -np.sin(phi) * np.sin(th), np.cos(phi)])
    south = -north
    west = np.array([np.sin(th), -np.cos(th), 0.0])
    ecl_pole = np.array([0.0, -np.sin(e), np.cos(e)])
    sense = 1.0 if lat >= 0 else -1.0
    out = {}
    for k in range(12):
        az = (270.0 - sense * 30.0 * k) * D2R                           # from the south point, towards the west
        d = np.cos(az) * south + np.sin(az) * west                       # horizontal direction of that azimuth
        n_vertical = np.cross(zenith, d)
        x = np.cross(n_vertical, ecl_pole)
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            return None
        x = x / norm
        if x @ d < 0.0:
            x = -x
        lon = np.arctan2(x[1] * np.cos(e) + x[2] * np.sin(e), x[0]) * R2D
        out[k + 1] = float(lon % 360.0)
    if abs((out[10] - mc + 180.0) % 360.0 - 180.0) > 90.0:               # the tenth cusp is the Midheaven itself
        out = {h: (v + 180.0) % 360.0 for h, v in out.items()}
    return out


def gauquelin_sectors(ramc_deg, eps_deg, lat_deg):
    """The 36 Gauquelin sector boundaries (ecliptic longitudes). Sector 1 starts at the Ascendant and the sectors
    run clockwise through the Midheaven side: 18 above the horizon and 18 below, each an equal share of the
    diurnal or nocturnal arc of the boundary point itself (a Placidus-style division). Raises ValueError where the
    division does not exist (inside the polar circles)."""
    import math
    e = math.radians(eps_deg)
    tan_phi = math.tan(math.radians(lat_deg))
    start = float(ascendant_from(ramc_deg, eps_deg, lat_deg))
    out = []
    for n in range(1, 37):
        lon = start + 10.0 * n if n > 1 else start
        for _ in range(2000):
            x = -tan_phi * math.tan(math.asin(math.sin(e) * math.sin(math.radians(lon))))
            if abs(x) >= 1.0:
                raise ValueError(f"Gauquelin sectors are undefined at latitude {lat_deg:.2f} (inside the polar circle)")
            dsa = math.degrees(math.acos(x))
            nsa = 180.0 - dsa
            h = -dsa + (n - 1) / 18.0 * 2.0 * dsa if n <= 18 else dsa + (n - 19) / 18.0 * 2.0 * nsa
            a = math.radians(ramc_deg - h)
            new = math.degrees(math.atan2(math.sin(a), math.cos(a) * math.cos(e))) % 360.0
            done = abs((new - lon + 180.0) % 360.0 - 180.0) < 1e-10
            lon = new
            if done:
                break
        out.append(float(lon))
    return out


def _porphyry(ramc, eps, lat, arcs, asc, mc):
    """Each quadrant of the ecliptic between the angles is divided into three equal parts."""
    start, direction, first = arcs[11]
    second = arcs[2][2]
    return {11: _norm360(start + direction * first / 3.0), 12: _norm360(start + direction * 2.0 * first / 3.0),
            2: _norm360(asc + direction * second / 3.0), 3: _norm360(asc + direction * 2.0 * second / 3.0)}


_THREE_QUADRANT = {
    "Placidus": _placidus, "Koch": _koch, "Regiomontanus": _regiomontanus, "Campanus": _campanus,
    "Alcabitius": _alcabitius, "Topocentric": _topocentric, "Morinus": _morinus, "Meridian": _meridian,
    "Porphyry": _porphyry, "Carter": _carter, "Krusinski": _krusinski, "Horizon": _horizon_houses, "Pullen SD": _pullen(_pullen_sd_outer), "Pullen SR": _pullen(_pullen_sr_outer),
}

# Swiss-style one-letter codes for people moving from that interface
LETTERS = {"G": "Gauquelin", "H": "Horizon", "U": "Krusinski", "L": "Pullen SD", "Q": "Pullen SR", "I": "Sunshine", "i": "Sunshine (Makransky)", "Y": "APC", "P": "Placidus", "K": "Koch", "O": "Porphyry", "R": "Regiomontanus", "C": "Campanus",
           "B": "Alcabitius", "T": "Topocentric", "M": "Morinus", "X": "Meridian", "E": "Equal", "A": "Equal",
           "W": "Whole sign", "V": "Vehlow", "D": "Equal (MC)", "N": "Equal (Aries)", "S": "Sripati",
           "F": "Carter"}


@dataclass
class Houses:
    cusps: list                      # 12 ecliptic longitudes, cusps[0] is house 1 (degrees)
    ascendant: float
    midheaven: float
    vertex: float
    ramc: float
    obliquity: float
    system: str
    notes: list = field(default_factory=list)


def compute(ramc_deg, eps_deg, lat_deg, system="Placidus", sun_dec=None):
    """House cusps for a right ascension of the Midheaven, true obliquity and latitude (all degrees)."""
    system = LETTERS.get(system, system)
    asc = float(ascendant_from(ramc_deg, eps_deg, lat_deg))
    mc0 = float(midheaven_from(ramc_deg, eps_deg))
    # Near the poles the meridian point the formula calls the Midheaven can lie below the horizon; the
    # tenth cusp is then the opposite point and the houses run clockwise.
    dec = np.radians(_dec_of_lon(mc0, eps_deg))
    phi = np.radians(lat_deg)
    alt = np.arcsin(np.clip(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec), -1.0, 1.0))
    mc_circle = mc0 if alt >= 0.0 else _norm360(mc0 + 180.0)
    # Systems built from great circles (Regiomontanus, Campanus) follow the geometry; the formula based
    # systems keep the plain Midheaven and counter-clockwise houses, as is the convention for them.
    circle_system = system in ("Regiomontanus", "Campanus", "Topocentric")
    mc = mc_circle if circle_system else mc0
    d = (asc - mc + 180.0) % 360.0 - 180.0
    if not circle_system:
        d = (asc - mc) % 360.0
    direction = 1.0 if d >= 0 else -1.0
    first = abs(d)
    arcs = {11: (mc, direction, first), 12: (mc, direction, first),
            2: (asc, direction, 180.0 - first), 3: (asc, direction, 180.0 - first)}
    notes = []
    cusp = {1: asc, 10: mc, 4: _norm360(mc + 180.0), 7: _norm360(asc + 180.0)}

    if system == "Equal":
        cusp = {k: _norm360(asc + 30.0 * (k - 1)) for k in range(1, 13)}
    elif system == "Equal (MC)":
        cusp = {k: _norm360(mc + 30.0 * (k - 10)) for k in range(1, 13)}
    elif system == "Equal (Aries)":
        cusp = {k: 30.0 * (k - 1) for k in range(1, 13)}
    elif system == "Vehlow":
        cusp = {k: _norm360(asc - 15.0 + 30.0 * (k - 1)) for k in range(1, 13)}
    elif system == "Whole sign":
        start = np.floor(asc / 30.0) * 30.0
        cusp = {k: _norm360(start + 30.0 * (k - 1)) for k in range(1, 13)}
    elif system in ("Sunshine", "Sunshine (Makransky)"):
        if sun_dec is None:
            raise ValueError("Sunshine houses need the Sun's declination")
        res = _sunshine(ramc_deg, eps_deg, lat_deg, sun_dec)
        if res is None:
            notes.append(f"Sunshine houses are undefined where the Sun does not rise or set; Porphyry houses used")
            res = _porphyry(ramc_deg, eps_deg, lat_deg, arcs, asc, mc)
            cusp.update({k: _norm360(v) for k, v in res.items()})
            for k, o in ((5, 11), (6, 12), (8, 2), (9, 3)):
                cusp[k] = _norm360(cusp[o] + 180.0)
            system = "Porphyry"
        else:
            cusp = {k: float(v) for k, v in res.items()}
    elif system == "APC":
        # APC (Ascendant Parallel Circle): Sunshine's construction on the daily circle of the Ascendant point instead of the Sun's
        dec_asc = float(np.degrees(np.arcsin(np.sin(eps_deg * D2R) * np.sin(asc * D2R))))
        res = _sunshine(ramc_deg, eps_deg, lat_deg, dec_asc)
        if res is None:
            notes.append("APC houses are undefined here (the Ascendant's circle never rises or sets); Porphyry houses used")
            res = _porphyry(ramc_deg, eps_deg, lat_deg, arcs, asc, mc)
            cusp.update({k: _norm360(v) for k, v in res.items()})
            for k, o in ((5, 11), (6, 12), (8, 2), (9, 3)):
                cusp[k] = _norm360(cusp[o] + 180.0)
            system = "Porphyry"
        else:
            cusp = {k: float(v) for k, v in res.items()}
    elif system == "Sripati":
        p = compute(ramc_deg, eps_deg, lat_deg, "Porphyry").cusps
        # Sripati cusps are the midpoints between consecutive Porphyry cusps
        mids = {k: _norm360(p[k - 2] + ((p[k - 1] - p[k - 2]) % 360.0) / 2.0) for k in range(1, 13)}
        cusp = mids
    elif system in _THREE_QUADRANT:
        res = _THREE_QUADRANT[system](ramc_deg, eps_deg, lat_deg, arcs, asc, mc)
        if res is None:
            notes.append(f"{system} is undefined at latitude {lat_deg:.2f} (circumpolar); Porphyry houses used")
            res = _porphyry(ramc_deg, eps_deg, lat_deg, arcs, asc, mc)
            system = "Porphyry"
        cusp.update({k: _norm360(v) for k, v in res.items()})
        if system not in ("Meridian", "Morinus", "Carter", "Krusinski", "Horizon"):
            for k, o in ((5, 11), (6, 12), (8, 2), (9, 3)):
                cusp[k] = _norm360(cusp[o] + 180.0)
    else:
        raise ValueError(f"unknown house system {system!r}")
    return Houses([float(cusp[k]) for k in range(1, 13)], asc, mc0,
                  float(vertex_from(ramc_deg, eps_deg, lat_deg)), float(ramc_deg % 360.0), float(eps_deg),
                  system, notes)


def houses(jd_ut, east_longitude_deg, latitude_deg, system="Placidus"):
    """House cusps and angles for a moment and place. Longitude is east-positive, latitude north-positive."""
    jd_tt = timescales.ut_to_tt(jd_ut)
    ramc = float(np.degrees(sidereal_time.local_apparent_sidereal_time(jd_ut, east_longitude_deg, jd_tt)))
    eps = float(np.degrees(frames.true_obliquity(jd_tt)))
    return compute(ramc, eps, latitude_deg, system)
