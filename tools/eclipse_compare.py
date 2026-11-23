import os
import sys, numpy as np, swisseph as real
import alif_ephemeris.swe_compat as ours
real.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
ours.set_ephe_path("data/raw")
fmt = lambda jd: "%.5f" % jd
print("LUNAR (next after start, then previous)")
for start in (2432050.0, 2451545.0, 2459000.5):
    for back in (False, True):
        a = ours.lun_eclipse_when(start, real.FLG_SWIEPH, 0, back); b = real.lun_eclipse_when(start, real.FLG_SWIEPH, 0, back)
        print(" start", start, "back" if back else "next", "kind ours/ref", a[0], b[0], "max diff %.1f s" % ((a[1][0] - b[1][0]) * 86400),
              "pen begin diff %.1f s, end %.1f s" % ((a[1][6] - b[1][1][6] if False else (a[1][6] - b[1][6]) * 86400), (a[1][7] - b[1][7]) * 86400),
              "partial begin diff %.1f s" % ((a[1][2] - b[1][2]) * 86400))
print("SOLAR GLOBAL")
for start in (2432050.0, 2451545.0, 2459000.5):
    for back in (False, True):
        a = ours.sol_eclipse_when_glob(start, real.FLG_SWIEPH, 0, back); b = real.sol_eclipse_when_glob(start, real.FLG_SWIEPH, 0, back)
        print(" start", start, "back" if back else "next", "kind ours/ref", a[0], b[0], "max diff %.1f s" % ((a[1][0] - b[1][0]) * 86400),
              "begin diff %.1f s, end %.1f s" % ((a[1][2] - b[1][2]) * 86400, (a[1][3] - b[1][3]) * 86400))
print("SOLAR LOCAL (Delhi)")
for start in (2432050.0, 2451545.0):
    for back in (False, True):
        place = (77.2167, 28.6667, 0.0)
        a = ours.sol_eclipse_when_loc(start, place, real.FLG_SWIEPH, back); b = real.sol_eclipse_when_loc(start, place, real.FLG_SWIEPH, back)
        print(" start", start, "back" if back else "next", "max diff %.1f s" % ((a[1][0] - b[1][0]) * 86400),
              "C1 diff %.1f s, C4 diff %.1f s" % ((a[1][1] - b[1][1]) * 86400, (a[1][4] - b[1][4]) * 86400))
