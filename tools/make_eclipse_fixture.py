"""Freeze reference eclipse circumstances (times in UT) so the tests can run without the reference program."""
import os
import json
import swisseph as real
real.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
out = {"note": "Reference lunar, solar and local eclipse times (UT Julian Day), 1950-2020, automatic Delta-T.", "lunar": [], "solar": [], "local": []}
for start in (2433000.5, 2438000.5, 2443000.5, 2448000.5, 2453000.5, 2457500.5):
    k, t = real.lun_eclipse_when(start, real.FLG_SWIEPH, 0, False)
    out["lunar"].append(dict(start=start, kind=k, tret=list(t[:8])))
    k, t = real.sol_eclipse_when_glob(start, real.FLG_SWIEPH, 0, False)
    out["solar"].append(dict(start=start, kind=k, tret=list(t[:4])))
    k, t, a = real.sol_eclipse_when_loc(start, (77.2167, 28.6667, 0.0), real.FLG_SWIEPH, False)
    out["local"].append(dict(start=start, kind=k, tret=list(t[:5]), place=[77.2167, 28.6667, 0.0]))
json.dump(out, open("tests/data/eclipse_reference.json", "w"), indent=1)
print("written")
