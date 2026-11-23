import os
import numpy as np, swisseph as real
from scipy.optimize import least_squares
import alif_ephemeris.swe_compat as ours
from alif_ephemeris import eclipses as E, timescales as T
real.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", "")); ours.set_ephe_path("data/raw")
spk = ours._spk()
starts = [2430000.5 + 1200 * k for k in range(0, 20)]
ref = [real.sol_eclipse_when_glob(s, real.FLG_SWIEPH, 0, False) for s in starts]
dt = lambda t: float(T.delta_t(t)) / 86400
def res(x):
    E.R_MOON_SOLAR, E.R_EARTH_SOLAR = x
    out = []
    for s, (kind, r) in zip(starts, ref):
        e = E.next_solar_eclipse(spk, s + dt(s), False)
        out += [(e.begin - dt(e.maximum) - r[2]) * 86400, (e.end - dt(e.maximum) - r[3]) * 86400]
    return np.array(out)
r0 = res([1737.4, 6378.14]); print("start rms %.1f worst %.1f" % (np.sqrt((r0**2).mean()), np.abs(r0).max()))
fit = least_squares(res, [1737.4, 6378.14], x_scale=[10, 10], diff_step=[1e-4, 1e-4])
r1 = res(fit.x); print("fit moon %.3f earth %.3f -> rms %.2f worst %.2f" % (fit.x[0], fit.x[1], np.sqrt((r1**2).mean()), np.abs(r1).max()))
print([ "%+.1f"%v for v in r1])
