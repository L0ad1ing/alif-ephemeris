"""Moon difference vs Horizons as a function of date (Horizons uses DE441 for the Moon; we use DE440)."""
import numpy as np
from alif_ephemeris import Ephemeris, julian_day, planets as P
import horizons_check as H
eph = Ephemeris("data/raw/de440s.bsp")
jds = julian_day(1850, 1, 1) + np.linspace(0, 365.25 * 295, 14)
ra, dec = H.horizons("301", jds, "1")
for jd, r, d in zip(jds, ra, dec):
    w = float(np.floor(jd - 0.5) + 0.5)
    v = P.astrometric_icrf(eph.spk, P.MOON, w, float(jd - w))
    o = v / np.linalg.norm(v)
    rr, dd = np.radians(r), np.radians(d)
    t = np.array([np.cos(dd) * np.cos(rr), np.cos(dd) * np.sin(rr), np.sin(dd)])
    sep = np.arctan2(np.linalg.norm(np.cross(o, t)), np.dot(o, t)) * 206264.806
    print("year %7.1f   Moon difference %.5f arcsec  (%.1f m at the Moon)" % (2000 + (jd - 2451545) / 365.25, sep, sep / 206264.806 * 384400e3))
