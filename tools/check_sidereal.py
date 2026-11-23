import os
import numpy as np, erfa, swisseph as swe
from alif_ephemeris import sidereal_time as S, timescales as T, julian_day
swe.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
A = 206264.806247
w = {"era": 0, "gmst": 0, "eqeq": 0, "gast": 0, "swiss_gast_deg": 0}
rng = np.random.default_rng(2)
for jd in julian_day(1850, 1, 1) + rng.uniform(0, 365.25 * 300, 300):
    tt = float(T.ut_to_tt(jd))
    ut1a, ut1b = T.split_jd(jd)
    tta, ttb = T.split_jd(tt)
    d = lambda a, b: abs((a - b + np.pi) % (2 * np.pi) - np.pi) * A * 1e6   # micro-arcsec
    w["era"] = max(w["era"], d(S.earth_rotation_angle(jd), erfa.era00(float(ut1a), float(ut1b))))
    w["gmst"] = max(w["gmst"], d(S.gmst(jd, tt), erfa.gmst06(float(ut1a), float(ut1b), float(tta), float(ttb))))
    w["eqeq"] = max(w["eqeq"], abs(S.equation_of_equinoxes(tt) - erfa.ee06a(float(tta), float(ttb))) * A * 1e6)
    w["gast"] = max(w["gast"], d(S.gast(jd, tt), erfa.gst06a(float(ut1a), float(ut1b), float(tta), float(ttb))))
    sw = swe.sidtime(jd) * 15.0
    w["swiss_gast_deg"] = max(w["swiss_gast_deg"], abs((sw - np.degrees(S.gast(jd, tt)) + 180) % 360 - 180) * 3600)
print("worst difference vs SOFA (micro-arcseconds):", {k: round(v, 4) for k, v in w.items() if k != "swiss_gast_deg"})
print("worst difference vs Swiss sidtime (arcseconds): %.5f" % w["swiss_gast_deg"])
