"""Save NASA JPL Horizons answers to tests/data/horizons_reference.json so the tests can run offline.

Needs network. Sends only a body id and a list of times (Terrestrial Time).
Run:  PYTHONPATH=src python tools/make_horizons_fixture.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import horizons_check as H                                   # noqa: E402
from alif_ephemeris import julian_day                        # noqa: E402

DATES_TT = [julian_day(1955, 3, 3, 6.5), julian_day(1985, 3, 3, 18.25), julian_day(2000, 1, 1, 12.0),
            julian_day(2010, 6, 15, 0.0), julian_day(2026, 11, 22, 12.0), julian_day(2027, 7, 4, 3.0),
            julian_day(2045, 12, 31, 23.5), julian_day(2088, 9, 9, 9.0)]
out = {"note": "Astrometric ICRF RA/Dec (degrees) from NASA JPL Horizons, geocentre, Terrestrial Time.",
       "fetched": "2026-11-22", "jd_tt": DATES_TT, "bodies": {}}
for ours_id, cmd, name in H.BODIES:
    ra, dec = H.horizons(cmd, DATES_TT, "1")
    out["bodies"][name] = {"id": ours_id, "ra": list(map(float, ra)), "dec": list(map(float, dec))}
    print(name, "ok")
os.makedirs("tests/data", exist_ok=True)
with open("tests/data/horizons_reference.json", "w") as f:
    json.dump(out, f, indent=1)
print("written")
