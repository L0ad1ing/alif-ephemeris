"""A small web API for Alif Ephemeris that you run yourself.

    python -m alif_ephemeris.server --file data/raw/de440.bsp --port 8765

It listens on your own computer (127.0.0.1) unless you say otherwise, uses only the Python standard library and
returns JSON. Nothing is sent anywhere. Example:

    http://127.0.0.1:8765/api/position?body=moon&jd=2461000.5
    http://127.0.0.1:8765/api/houses?jd=2461000.5&lon=-77&lat=39.4&system=P
    http://127.0.0.1:8765/api/ayanamsa?name=Lahiri%20(JHora)&jd=2461000.5
    http://127.0.0.1:8765/api/star?name=Spica&jd=2461000.5
    http://127.0.0.1:8765/api/riseset?body=sun&jd=2461000.5&lon=-77&lat=39.4
    http://127.0.0.1:8765/api/eclipse?kind=lunar&jd=2461000.5
    http://127.0.0.1:8765/api/health

Times: `jd` is a Julian Day in UT. Add `date=2026-11-22T12:00:00` instead if you prefer a calendar date (UT).
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import ayanamsa as _ay
from . import houses as _houses
from . import swe_compat as swe
from . import timescales
from .ephemeris import Ephemeris
from .swe_compat import Error

BODIES = {"sun": swe.SUN, "moon": swe.MOON, "mercury": swe.MERCURY, "venus": swe.VENUS, "mars": swe.MARS,
          "jupiter": swe.JUPITER, "saturn": swe.SATURN, "uranus": swe.URANUS, "neptune": swe.NEPTUNE, "pluto": swe.PLUTO,
          "mean node": swe.MEAN_NODE, "true node": swe.TRUE_NODE, "mean apogee": swe.MEAN_APOG,
          "osculating apogee": swe.OSCU_APOG, "chiron": swe.CHIRON, "pholus": swe.PHOLUS, "ceres": swe.CERES,
          "pallas": swe.PALLAS, "juno": swe.JUNO, "vesta": swe.VESTA, "cupido": swe.CUPIDO, "hades": swe.HADES,
          "zeus": swe.ZEUS, "kronos": swe.KRONOS, "apollon": swe.APOLLON, "admetos": swe.ADMETOS,
          "vulkanus": swe.VULKANUS, "poseidon": swe.POSEIDON, "transpluto": swe.ISIS, "white moon": swe.WHITE_MOON,
          "proserpina": swe.PROSERPINA}


def _jd(q):
    if "jd" in q:
        return float(q["jd"][0])
    if "date" in q:
        d = q["date"][0]
        date, _, clock = d.partition("T")
        y, m, dd = (int(x) for x in date.split("-"))
        h = mi = s = 0.0
        if clock:
            parts = clock.rstrip("Z").split(":")
            h = float(parts[0]); mi = float(parts[1]) if len(parts) > 1 else 0.0; s = float(parts[2]) if len(parts) > 2 else 0.0
        return swe.julday(y, m, dd, h + mi / 60.0 + s / 3600.0)
    raise ValueError("give jd=<Julian Day UT> or date=YYYY-MM-DDTHH:MM:SS")


def _flag(q, name):
    return q.get(name, ["0"])[0].lower() in ("1", "true", "yes")


def handle(path: str, query: dict) -> dict:
    """Pure function behind every route; used by the HTTP handler and by the tests."""
    if path == "/api/health":
        lo, hi = swe._spk().coverage()
        return {"status": "ok", "covers_jd": [lo, hi], "ayanamsas": len(_ay.names()), "house_systems": sorted(set(_houses.LETTERS.values()))}
    if path == "/api/position":
        name = query.get("body", ["sun"])[0].lower()
        if name not in BODIES:
            raise ValueError("unknown body; try one of: " + ", ".join(sorted(BODIES)))
        flags = swe.FLG_SWIEPH | swe.FLG_SPEED
        sidereal = query.get("sidereal", [None])[0]
        jd = _jd(query)
        if sidereal:
            pos, _ = swe.calc_ut(jd, BODIES[name], flags)
            tt = jd + swe.deltat(jd)
            ay = float(_ay.ayanamsa(sidereal, tt, true=True))
            lon = (pos[0] - ay) % 360.0
        else:
            pos, _ = swe.calc_ut(jd, BODIES[name], flags)
            lon, ay = pos[0], 0.0
        return {"body": name, "jd_ut": jd, "longitude": lon, "latitude": pos[1], "distance_au": pos[2],
                "longitude_speed": pos[3], "ayanamsa": ay, "sidereal": sidereal}
    if path == "/api/houses":
        h = _houses.houses(_jd(query), float(query["lon"][0]), float(query["lat"][0]), query.get("system", ["Placidus"])[0])
        return {"system": h.system, "cusps": h.cusps, "ascendant": h.ascendant, "midheaven": h.midheaven,
                "vertex": h.vertex, "ramc": h.ramc, "notes": h.notes}
    if path == "/api/ayanamsa":
        if "name" not in query:
            return {"names": _ay.names()}
        tt = _jd(query) + swe.deltat(_jd(query))
        return {"name": query["name"][0], "jd_ut": _jd(query), "mean": float(_ay.ayanamsa(query["name"][0], tt)),
                "true": float(_ay.ayanamsa(query["name"][0], tt, true=True))}
    if path == "/api/star":
        jd = _jd(query)
        res = swe.fixstar_ut(query["name"][0], jd, swe.FLG_SWIEPH)
        return {"name": res[1], "jd_ut": jd, "longitude": res[0][0], "latitude": res[0][1]}
    if path == "/api/riseset":
        body = query.get("body", ["sun"])[0].lower()
        target = BODIES[body] if body in BODIES else query["body"][0]
        place = (float(query["lon"][0]), float(query["lat"][0]), float(query.get("height", ["0"])[0]))
        jd = _jd(query)
        out = {"body": body, "from_jd_ut": jd}
        for label, bit in (("rise", swe.CALC_RISE), ("set", swe.CALC_SET), ("upper_transit", swe.CALC_MTRANSIT)):
            code, t = swe.rise_trans(jd, target, bit, place)
            out[label] = t[0] if code == 0 else None
        return out
    if path == "/api/eclipse":
        jd, back = _jd(query), _flag(query, "back")
        kind = query.get("kind", ["lunar"])[0]
        if kind == "lunar":
            flag, t = swe.lun_eclipse_when(jd, swe.FLG_SWIEPH, 0, back)
        elif kind == "solar":
            flag, t = swe.sol_eclipse_when_glob(jd, swe.FLG_SWIEPH, 0, back)
        else:
            raise ValueError("kind must be lunar or solar")
        return {"kind": kind, "type_flags": flag, "maximum_jd_ut": t[0], "times_jd_ut": list(t[:8])}
    raise KeyError(path)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):          # quiet
        pass

    def do_GET(self):                   # noqa: N802
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)
        try:
            body, code = handle(url.path, query), 200
        except KeyError:
            body, code = {"error": "unknown route", "routes": ["/api/health", "/api/position", "/api/houses", "/api/ayanamsa", "/api/star", "/api/riseset", "/api/eclipse"]}, 404
        except (ValueError, Error, IndexError) as exc:
            body, code = {"error": str(exc)}, 400
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


def make_server(host="127.0.0.1", port=8765):
    return ThreadingHTTPServer((host, port), _Handler)


def main():
    ap = argparse.ArgumentParser(description="Alif Ephemeris web API (run it yourself)")
    ap.add_argument("--file", help="path to a JPL DE440 .bsp file (or set ALIF_EPHEMERIS_FILE)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.file:
        os.environ["ALIF_EPHEMERIS_FILE"] = a.file
    srv = make_server(a.host, a.port)
    print(f"Alif Ephemeris API on http://{a.host}:{a.port}/api/health  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
