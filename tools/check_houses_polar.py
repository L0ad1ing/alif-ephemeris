import os
import numpy as np, swisseph as swe
from alif_ephemeris import houses as H, julian_day
swe.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
rng = np.random.default_rng(9)
d = lambda a, b: abs((a - b + 180) % 360 - 180) * 3600
for sysl in "PKOCRBTEWMX":
    name = H.LETTERS[sysl]
    worst = 0; fb_mismatch = 0; n = 0; swiss_err = 0; our_fb = 0
    for _ in range(300):
        jd = julian_day(1900, 1, 1) + rng.uniform(0, 365.25 * 150)
        lo = rng.uniform(-180, 180)
        la = rng.choice([-1, 1]) * rng.uniform(60, 89.99)
        try:
            cusps, ascmc = swe.houses_ex(jd, la, lo, sysl.encode())
            sw_ok = True
        except Exception:
            sw_ok = False; swiss_err += 1; continue
        ours = H.houses(jd, lo, la, name)
        if ours.system != name: our_fb += 1
        n += 1
        w = max(d(ours.cusps[k], cusps[k]) for k in range(12))
        if w > 0.05: fb_mismatch += 1
        worst = max(worst, min(w, 1e4))
    print("%-14s tested %3d  mismatches over 0.05 arcsec: %3d  (we fell back %3d times, Swiss raised %d)" % (name, n, fb_mismatch, our_fb, swiss_err))
