# Alif Ephemeris: rebuild guide

How to rebuild this project from nothing. No secrets are used. The project needs no accounts.

1. Install Python 3.10 or newer. Clone the repo (or copy the folder).
2. `pip install -e ".[dev]"` (numpy and scipy for the library; pytest, skyfield, jplephem, pyerfa, pyswisseph for the tests).
3. Download the NASA file listed in `docs/DATA.md` into `data/raw/`.
4. `python -m pytest -q` (about 100 tests, all should pass; a few skip if the optional data files are missing).
5. `PYTHONPATH=src python tools/horizons_check.py [folder with Swiss .se1 files]` compares us (and optionally Swiss)
   with NASA Horizons. `tools/accuracy_report.py` compares with Skyfield and SOFA. These are checks, not part of the library.
6. `planet_centres.npz` is rebuilt by `tools/build_planet_centres.py` (needs network). The other small data files in `src/alif_ephemeris/data/` are committed. To regenerate the nutation table run
   `python tools/export_nutation_tables.py` (needs Skyfield installed).

7. Optional, for dates outside 1550 to 2650: the two DE441 files from `docs/DATA.md` next to `de440.bsp`.
8. To run Alif Astrology Software's own tests on this library: `tools/alif_shim_plugin.py` (see `docs/SWISS_COMPATIBILITY.md`).

Where things are: `docs/PLAN.md` (what is done, what is next), `docs/CLEAN_ROOM.md` (hard rules),
`CLAUDE.md` (working rules), `NOTICE` (credit).

Gotchas
- Never add a small number to a full float Julian Day. Use (whole, fraction) time (see `planets.py`).
- Compare tiny angles with atan2(|cross|, dot). `arccos` loses precision below 3 milli-arcseconds.
- On Windows shells, long scripts with apostrophes break heredocs. Put test scripts in files.
