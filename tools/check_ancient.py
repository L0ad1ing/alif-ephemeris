"""Compare Alif Ephemeris with the reference program at historical and far-future dates (needs the DE441 files)."""
import os
import numpy as np
import swisseph as real
import alif_ephemeris.swe_compat as ours

real.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
ours.set_ephe_path("data/raw")
print("data files used:", [os.path.basename(f) for f in ours._data_files()])
d = lambda a, b: abs((a - b + 180) % 360 - 180) * 3600
for year in (-3000, -1000, -500, 0, 500, 1000, 1400, 1500, 2700, 3000, 5000):
    jd = real.julday(year, 6, 1, 12.0)
    row = []
    for b, n in ((0, "Sun"), (1, "Moon"), (2, "Mercury"), (4, "Mars"), (5, "Jupiter"), (6, "Saturn")):
        try:
            a = ours.calc_ut(jd, b, real.FLG_SWIEPH)[0]
            r = real.calc_ut(jd, b, real.FLG_SWIEPH)[0]
            row.append("%s %.1f" % (n, d(a[0], r[0])))
        except Exception as e:
            row.append("%s ERR(%s)" % (n, str(e)[:22]))
    print("year %5d (arcsec vs reference): %s" % (year, "  ".join(row)))
