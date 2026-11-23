# Alif Ephemeris

A free, open-source astronomical ephemeris for astrology and astronomy software. Goal: everything
Swiss Ephemeris offers, plus more, free for anyone including commercial use. Apache-2.0.
Credit required: "Alif Ephemeris by AlifAstrology.com" (kept in NOTICE). Help: info@alifastrology.com.

Read `docs/PLAN.md` first (what is built, what is next), then `docs/CLEAN_ROOM.md` (hard rule).

## Hard rules
- Clean room. Never read, port or copy Swiss Ephemeris code, data files or documentation text.
  Swiss (pyswisseph) is a test oracle only, never imported by `src/`.
- Truth is JPL's own data (via DE440/DE441). Report the max error of every result we can.
- Pure functions, no global state, thread-safe. No hidden file handles that stay open per call.
- Dates: Julian Day numbers as floats in `src/`; no clock inside the library.
- Plain words in docs. Short sentences. No em-dashes in user-facing text.
- Keep REBUILD.md current (standing rule: every project gets one).

## Commands
`python -m pytest -q` runs the tests. `python tools/accuracy_report.py` prints accuracy tables.
Raw JPL files live in `data/raw/` (gitignored, download links in docs/DATA.md).
