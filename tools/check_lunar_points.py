import os
import numpy as np, swisseph as swe
from alif_ephemeris import Ephemeris, lunar_points as L, julian_day, timescales as T, frames
swe.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
e = Ephemeris("data/raw/de440s.bsp")
rng = np.random.default_rng(6)
dates = julian_day(1900, 1, 1) + rng.uniform(0, 365.25 * 150, 40)
d = lambda a, b: abs((a - b + 180) % 360 - 180) * 3600
w = dict(mean_node_plain=0, mean_node_true_equinox=0, true_node=0, mean_apogee_plain=0, mean_apogee_nut=0, osc_apogee=0)
for jd in dates:
    tt = float(T.ut_to_tt(jd)); whole, frac = T.split_jd(tt); whole, frac = float(whole), float(frac)
    dpsi = float(np.degrees(frames.nutation_iau2000a(tt)[0]))
    mn = float(L.mean_node(tt)); ma = float(L.mean_apogee(tt))
    sn = swe.calc(tt, swe.MEAN_NODE, swe.FLG_SWIEPH)[0][0]
    sa = swe.calc(tt, swe.MEAN_APOG, swe.FLG_SWIEPH)[0][0]
    st = swe.calc(tt, swe.TRUE_NODE, swe.FLG_SWIEPH)[0][0]
    so = swe.calc(tt, swe.OSCU_APOG, swe.FLG_SWIEPH)[0][0]
    w["mean_node_plain"] = max(w["mean_node_plain"], d(mn, sn))
    w["mean_node_true_equinox"] = max(w["mean_node_true_equinox"], d(mn + dpsi, sn))
    w["mean_apogee_plain"] = max(w["mean_apogee_plain"], d(ma, sa))
    w["mean_apogee_nut"] = max(w["mean_apogee_nut"], d(ma + dpsi, sa))
    w["true_node"] = max(w["true_node"], d(float(L.true_node(e.spk, whole, frac)), st))
    w["osc_apogee"] = max(w["osc_apogee"], d(float(L.osculating_apogee(e.spk, whole, frac)[0]), so))
for k, v in w.items(): print("%-26s worst %.4f arcsec" % (k, v))
