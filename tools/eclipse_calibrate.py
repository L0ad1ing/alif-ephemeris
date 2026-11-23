"""Fit the shadow-radius constants of eclipses.py to reference contact times (development tool)."""
import os
import numpy as np, swisseph as real
from scipy.optimize import least_squares
import alif_ephemeris.swe_compat as ours
from alif_ephemeris import eclipses as E, timescales as T
real.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", "")); ours.set_ephe_path("data/raw")
spk = ours._spk()
starts = [2430000.5 + 2500 * k for k in range(0, 16)]
ref = [real.lun_eclipse_when(s, real.FLG_SWIEPH, 0, False) for s in starts]
dt = lambda t: float(T.delta_t(t)) / 86400
def residuals(x):
    E.UMBRA_ENLARGEMENT, E.PENUMBRA_ENLARGEMENT, E.R_MOON = x[0], x[1], x[2]; E.R_SUN = 696000.0
    out = []
    for s, (kind, r) in zip(starts, ref):
        e = E.next_lunar_eclipse(spk, s + dt(s), False)
        out += [(e.penumbral[0] - dt(e.maximum) - r[6]) * 86400, (e.penumbral[1] - dt(e.maximum) - r[7]) * 86400]
        if e.partial and r[2]: out += [(e.partial[0] - dt(e.maximum) - r[2]) * 86400, (e.partial[1] - dt(e.maximum) - r[3]) * 86400]
        if e.total and r[4]: out += [(e.total[0] - dt(e.maximum) - r[4]) * 86400, (e.total[1] - dt(e.maximum) - r[5]) * 86400]
    return np.array(out)
r0 = residuals([1.0, 1.0, 1737.4])
fit = least_squares(residuals, [1.0, 1.0, 1737.4], x_scale=[0.01, 0.01, 10], diff_step=[1e-4, 1e-4, 1e-4])
E.UMBRA_ENLARGEMENT, E.PENUMBRA_ENLARGEMENT, E.R_MOON = fit.x
print("umbra %.5f penumbra %.5f moon %.3f" % tuple(fit.x))
print("%-22s %-8s %-9s %-9s %-9s %-9s %-9s %-9s %s" % ("start", "kind", "max", "pen beg", "pen end", "par beg", "par end", "tot beg", "z/Moon dist"))
for s_, (kind, r) in zip(starts, ref):
    e = E.next_lunar_eclipse(spk, s_ + dt(s_), False)
    d = lambda a, b: "%+.1f" % ((a - dt(e.maximum) - b) * 86400) if (a and b) else "-"
    print("%-22.1f %-8d %-9s %-9s %-9s %-9s %-9s %-9s" % (s_, kind, d(e.maximum, r[0]), d(e.penumbral[0], r[6]), d(e.penumbral[1], r[7]),
          d(e.partial[0], r[2]) if e.partial else "-", d(e.partial[1], r[3]) if e.partial else "-", d(e.total[0], r[4]) if e.total else "-"))
