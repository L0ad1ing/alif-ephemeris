"""Compare our planet-centre positions with Swiss Ephemeris, both in barycentre and centre-of-body modes."""
import sys
import numpy as np
import swisseph as swe
from alif_ephemeris import Ephemeris, julian_day
from alif_ephemeris import planets as P

swe.set_ephe_path(sys.argv[1])
print("has CENTER_BODY flag:", hasattr(swe, "FLG_CENTER_BODY"), getattr(swe, "FLG_CENTER_BODY", None), " lib", swe.version if hasattr(swe, "version") else "")
eph = Ephemeris("data/raw/de440s.bsp")
rng = np.random.default_rng(11)
dates = julian_day(1950, 1, 1) + rng.uniform(0, 365.25 * 100, 40)
names = {5: ("Jupiter", P.JUPITER, 5), 6: ("Saturn", P.SATURN, 6), 7: ("Uranus", P.URANUS, 7), 8: ("Neptune", P.NEPTUNE, 8), 9: ("Pluto", P.PLUTO, 9)}


def wrap(d):
    return (d + 180) % 360 - 180


print("%-8s %-24s %-24s %-24s" % ("", "ours(centre) - Swiss(default)", "ours(centre) - Swiss(COB)", "ours(bary) - Swiss(default)"))
for swid, (name, ours_id, bary) in names.items():
    d1 = d2 = d3 = 0.0
    for jd in dates:
        tt = float(__import__("alif_ephemeris").timescales.ut_to_tt(jd))
        c = eph.apparent(ours_id, jd_tt=tt, speed=False).longitude
        b = eph.apparent(bary, jd_tt=tt, speed=False).longitude
        s0 = swe.calc(tt, swid, swe.FLG_SWIEPH)[0][0]
        d1 = max(d1, abs(wrap(c - s0)) * 3600)
        d3 = max(d3, abs(wrap(b - s0)) * 3600)
        try:
            s1 = swe.calc(tt, swid, swe.FLG_SWIEPH | swe.FLG_CENTER_BODY)[0][0]
            d2 = max(d2, abs(wrap(c - s1)) * 3600)
        except Exception as e:
            d2 = float("nan")
    print("%-8s %-24.5f %-24.5f %-24.5f" % (name, d1, d2, d3))
