"""Compare our house cusps and angles with Swiss Ephemeris (black-box oracle, development only)."""
import os
import sys
import numpy as np
import swisseph as swe
from alif_ephemeris import houses as H, julian_day

swe.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
rng = np.random.default_rng(4)
jds = julian_day(1900, 1, 1) + rng.uniform(0, 365.25 * 150, 60)
lons = rng.uniform(-180, 180, 60)
lats = rng.uniform(-60, 60, 60)
systems = sys.argv[1:] or list("PKORCBTMXEWVDNS")
d = lambda a, b: abs((a - b + 180) % 360 - 180) * 3600
for sysletter in systems:
    name = H.LETTERS[sysletter]
    worst = np.zeros(12)
    angles = np.zeros(3)
    bad = 0
    for jd, lo, la in zip(jds, lons, lats):
        try:
            cusps, ascmc = swe.houses_ex(jd, la, lo, sysletter.encode())
        except Exception as e:
            bad += 1
            continue
        ours = H.houses(jd, lo, la, name)
        if ours.system != name:      # we fell back; Swiss may or may not
            bad += 1
            continue
        for k in range(12):
            worst[k] = max(worst[k], d(ours.cusps[k], cusps[k]))
        angles = np.maximum(angles, [d(ours.ascendant, ascmc[0]), d(ours.midheaven, ascmc[1]), d(ours.vertex, ascmc[3])])
    print("%-12s %s  asc/mc/vertex: %s   skipped %d" % (name, "cusps worst (arcsec): " + " ".join("%.4f" % w for w in worst), " ".join("%.4f" % a for a in angles), bad))
