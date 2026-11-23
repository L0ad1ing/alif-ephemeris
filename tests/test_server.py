"""The self-hosted web API."""
import json
import os
import threading
import urllib.request

import pytest

DE440 = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "de440.bsp")
pytestmark = pytest.mark.skipif(not os.path.exists(DE440), reason="data/raw/de440.bsp not downloaded")


@pytest.fixture(scope="module")
def api():
    os.environ["ALIF_EPHEMERIS_FILE"] = os.path.abspath(DE440)
    from alif_ephemeris import server
    srv = server.make_server("127.0.0.1", 0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield "http://127.0.0.1:%d" % port
    srv.shutdown()


def get(base, route):
    try:
        with urllib.request.urlopen(base + route, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_and_position(api):
    code, data = get(api, "/api/health")
    assert code == 200 and data["status"] == "ok" and data["ayanamsas"] >= 40
    code, data = get(api, "/api/position?body=sun&jd=2451545.0")
    assert code == 200 and abs(data["longitude"] - 280.3689) < 0.002


def test_houses_ayanamsa_star_riseset_eclipse(api):
    code, data = get(api, "/api/houses?jd=2451545.0&lon=-77&lat=39.4&system=P")
    assert code == 200 and len(data["cusps"]) == 12
    code, data = get(api, "/api/ayanamsa?name=Lahiri%20(JHora)&jd=2451545.0")
    assert code == 200 and abs(data["mean"] - 23.84136) < 0.001
    code, data = get(api, "/api/star?name=Spica&jd=2451545.0")
    assert code == 200 and 203.8 < data["longitude"] < 204.0
    code, data = get(api, "/api/riseset?body=sun&jd=2451545.0&lon=0&lat=51.5")
    assert code == 200 and data["rise"] and data["set"]
    code, data = get(api, "/api/eclipse?kind=lunar&jd=2451545.0")
    assert code == 200 and data["maximum_jd_ut"] > 2451545.0


def test_errors_are_plain_json(api):
    code, data = get(api, "/api/position?body=nothing&jd=2451545.0")
    assert code == 400 and "unknown body" in data["error"]
    code, data = get(api, "/api/nothing")
    assert code == 404 and "routes" in data
