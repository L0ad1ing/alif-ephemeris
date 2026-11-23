import os
import numpy as np, swisseph as swe
from alif_ephemeris import ayanamsa as A, timescales as T, julian_day
swe.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
J2000 = 2451545.0
modes = sorted((getattr(swe, n), n[5:]) for n in dir(swe) if n.startswith("SIDM_") and n != "SIDM_USER")
rng = np.random.default_rng(1)
near = julian_day(1800, 1, 1) + rng.uniform(0, 365.25 * 400, 40)
wide = julian_day(-1000, 1, 1) + rng.uniform(0, 365.25 * 4000, 40)
print("%-4s %-24s %10s   %-22s %-22s" % ("id", "name", "J2000 deg", "1800-2200 worst (arcsec)", "-1000..3000 worst"))
for mid, name in modes:
    swe.set_sid_mode(mid, 0, 0)
    a0 = swe.get_ayanamsa_ex(J2000, swe.FLG_SWIEPH | swe.FLG_NONUT)[1]
    res = []
    for dates in (near, wide):
        w = 0
        for jd in dates:
            tt = float(T.ut_to_tt(jd))
            ref = swe.get_ayanamsa_ex(tt, swe.FLG_SWIEPH | swe.FLG_NONUT)[1]
            mine = float(A.fixed_point_ayanamsa(a0, J2000, tt))
            w = max(w, abs((ref - mine + 180.0) % 360.0 - 180.0) * 3600)
        res.append(w)
    print("%-4d %-24s %10.5f   %-22.4f %-22.3f" % (mid, name, a0, res[0], res[1]))
