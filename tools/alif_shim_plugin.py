"""pytest plugin: run Alif Astrology Software's own tests on Alif Ephemeris instead of Swiss Ephemeris.

It installs alif_ephemeris.swe_compat under the name `swisseph` before Alif is imported, so Alif's code and
tests run unchanged.

Usage (from the Alif software folder):
    PYTHONPATH="<path to Alif-Ephemeris>/src;<path to Alif-Ephemeris>/tools" \
        python -m pytest -p alif_shim_plugin tests/test_western.py
"""
import os
import sys

os.environ.setdefault("ALIF_EPHEMERIS_FILE", "data/raw/de440.bsp")
import alif_ephemeris.swe_compat as _compat  # noqa: E402

sys.modules["swisseph"] = _compat


def pytest_report_header(config):
    return "swisseph replaced by alif_ephemeris.swe_compat (data file %s)" % os.environ["ALIF_EPHEMERIS_FILE"]
