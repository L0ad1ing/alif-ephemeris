"""Compare mean and true node (longitude, speed) with the reference at a range of dates."""
import os
import numpy as np
import swisseph as real
import alif_ephemeris.swe_compat as ours

real.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
ours.set_ephe_path("data/raw")
d = lambda a, b: (a - b + 180) % 360 - 180
for flags in (real.FLG_SWIEPH, real.FLG_SWIEPH | real.FLG_SPEED, real.FLG_SWIEPH | real.FLG_SIDEREAL | real.FLG_SPEED):
    for sid in (real.SIDM_LAHIRI,):
        real.set_sid_mode(sid); ours.set_sid_mode(sid)
        for body, n in ((real.MEAN_NODE, "mean"), (real.TRUE_NODE, "true")):
            diffs = []
            for jd in np.linspace(2415020.5, 2470000.5, 12):
                a = ours.calc_ut(jd, body, flags)[0]
                b = real.calc_ut(jd, body, flags)[0]
                diffs.append(d(a[0], b[0]) * 3600)
            print("flags %d %s: arcsec diffs min %.2f max %.2f" % (flags, n, min(diffs), max(diffs)))
