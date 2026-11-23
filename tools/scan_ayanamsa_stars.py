"""Compare every sidereal mode and a list of fixed stars with the reference program (test oracle)."""
import os
import numpy as np
import swisseph as real
import alif_ephemeris.swe_compat as ours

real.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
ours.set_ephe_path("data/raw")
jds = [real.julday(y, 6, 1, 12.0) for y in (1900, 1950, 2000, 2026)]
print("== ayanamsa modes: difference (arcsec) at 1900, 1950, 2000, 2026")
bad = []
for mode in range(0, 47):
    try:
        real.set_sid_mode(mode)
        a_ref = [real.get_ayanamsa_ut(j) for j in jds]
    except Exception:
        continue
    try:
        ours.set_sid_mode(mode)
        a_our = [ours.get_ayanamsa_ut(j) for j in jds]
    except Exception as exc:
        print("mode %2d NOT AVAILABLE in ours: %s" % (mode, str(exc)[:60]))
        continue
    d = [(x - y) * 3600 for x, y in zip(a_our, a_ref)]
    flag = "" if max(abs(v) for v in d) < 2.0 else "  <-- differs"
    if flag:
        bad.append((mode, d))
    print("mode %2d  %s%s" % (mode, "  ".join("%+9.2f" % v for v in d), flag))
print("\n== fixed stars: longitude/latitude difference (arcsec) at 2000 and 2026")
names = ["Spica", "Regulus", "Aldebaran", "Antares", "Sirius", "Algol", "Arcturus", "Vega", "Fomalhaut", "Pollux", "Capella",
         "Betelgeuse", "Rigel", "Procyon", "Altair", "Canopus", "Alcyone", "Polaris", "Deneb", "Achernar"]
for n in names:
    row = []
    for j in (real.julday(2000, 1, 1, 12.0), real.julday(2026, 6, 1, 12.0)):
        try:
            a = ours.fixstar_ut(n, j, real.FLG_SWIEPH)[0]
            b = real.fixstar_ut(n, j, real.FLG_SWIEPH)[0]
            row.append("%7.2f/%6.2f" % (((a[0] - b[0] + 180) % 360 - 180) * 3600, (a[1] - b[1]) * 3600))
        except Exception as exc:
            row.append("ERR %s" % str(exc)[:30])
    print("%-11s %s" % (n, "   ".join(row)))
