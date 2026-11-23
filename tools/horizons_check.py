"""Compare our apparent positions with NASA JPL Horizons, the official JPL service.

Horizons uses the satellite ephemerides, so it gives true planet centres for Jupiter to Pluto.
Sends only a body id and a list of times (Terrestrial Time). Development check; needs network.

Run:  PYTHONPATH=src python tools/horizons_check.py
"""
import json
import urllib.parse
import urllib.request

import numpy as np

from alif_ephemeris import Ephemeris, julian_day, timescales as T
from alif_ephemeris import SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO

BODIES = [(SUN, "10", "Sun"), (MOON, "301", "Moon"), (MERCURY, "199", "Mercury"), (VENUS, "299", "Venus"),
          (MARS, "499", "Mars"), (JUPITER, "599", "Jupiter"), (SATURN, "699", "Saturn"),
          (URANUS, "799", "Uranus"), (NEPTUNE, "899", "Neptune"), (PLUTO, "999", "Pluto")]


def horizons(command, jds_tt, quantity):
    params = {
        "format": "json", "COMMAND": "'%s'" % command, "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'", "CENTER": "'500@399'", "TLIST": "'" + " ".join("%.9f" % j for j in jds_tt) + "'",
        "TIME_TYPE": "'TT'", "QUANTITIES": "'%s'" % quantity, "ANG_FORMAT": "'DEG'", "CSV_FORMAT": "'YES'",
        "EXTRA_PREC": "'YES'", "APPARENT": "'AIRLESS'",
    }
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        text = json.loads(r.read())["result"]
    block = text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
    a, b = [], []
    for line in block:
        parts = [p.strip() for p in line.split(",")]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                pass
        a.append(nums[-2])
        b.append(nums[-1])
    return np.array(a), np.array(b)


def wrap(d):
    return (d + 180) % 360 - 180


def main():
    import sys
    from alif_ephemeris import planets as P
    swe = None
    if len(sys.argv) > 1:
        import swisseph as swe
        swe.set_ephe_path(sys.argv[1])
    eph = Ephemeris("data/raw/de440s.bsp")
    rng = np.random.default_rng(5)
    jds = julian_day(1950, 1, 1) + np.sort(rng.uniform(0, 365.25 * 100, 20))   # 1950 to 2050, TT
    print("Astrometric ICRF place vs NASA JPL Horizons (no precession model involved)")
    print("angular separation, worst of %d dates 1950-2050, in arcseconds" % len(jds))
    print("%-8s %16s %16s" % ("Body", "Alif Ephemeris", "Swiss Ephemeris"))
    swid = {SUN: 0, MOON: 1, MERCURY: 2, VENUS: 3, MARS: 4, JUPITER: 5, SATURN: 6, URANUS: 7, NEPTUNE: 8, PLUTO: 9}
    flags = 0
    if swe:
        flags = swe.FLG_SWIEPH | swe.FLG_ICRS | swe.FLG_J2000 | swe.FLG_NOABERR | swe.FLG_NOGDEFL | swe.FLG_EQUATORIAL
    for ours_id, cmd, name in BODIES:
        ra, dec = horizons(cmd, jds, "1")
        worst = worst_sw = 0.0
        for jd, r, d in zip(jds, ra, dec):
            whole = float(np.floor(jd - 0.5) + 0.5)
            v = P.astrometric_icrf(eph.spk, ours_id, whole, float(jd - whole))
            ours = v / np.linalg.norm(v)
            rr, dd = np.radians(r), np.radians(d)
            theirs = np.array([np.cos(dd) * np.cos(rr), np.cos(dd) * np.sin(rr), np.sin(dd)])
            worst = max(worst, np.arctan2(np.linalg.norm(np.cross(ours, theirs)), np.dot(ours, theirs)) * 206264.806)
            if swe:
                x = swe.calc(float(jd), swid[ours_id], flags)[0]
                sr, sd = np.radians(x[0]), np.radians(x[1])
                sv = np.array([np.cos(sd) * np.cos(sr), np.cos(sd) * np.sin(sr), np.sin(sd)])
                worst_sw = max(worst_sw, np.arctan2(np.linalg.norm(np.cross(sv, theirs)), np.dot(sv, theirs)) * 206264.806)
        print("%-8s %16.5f %16s" % (name, worst, "%.5f" % worst_sw if swe else "-"))


if __name__ == "__main__":
    main()
