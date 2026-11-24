# Changelog

## 1.1.0

Speed
- Planet positions with speed, houses, azimuth and altitude, sidereal time, fixed stars and the ayanamsa now run at about the
  same speed as Swiss Ephemeris once warm (4 to 8 microseconds a call). `python tools/bench_speed.py` prints the comparison.
- Rise and set searches take about 130 microseconds (they took 45 milliseconds).
- Optional compiled helper for Placidus houses (`tools/build_core.bat` on Windows); plain Python is used without it.
- Loading the ephemeris no longer reads the whole data file into memory (540 MB down to about 100 MB with Python and numpy).

Data
- Compact ephemeris files (`tools/make_compact.py`, `alif_ephemeris.compact`): about a fifth of the size, expanded once into
  ordinary JPL files; positions change by under 0.0004 arcsecond (0.02 arcsecond for the true node).
- Range packs cut straight from DE441 (`tools/make_range_pack.py`) cover the years -1000 to 3200 beyond DE440.

Swiss-style interface
- New: heliacal events and the limiting magnitude (simple classical model), `csroundsec`, the model and tidal constants.
- New: Saros series and member numbers for solar and lunar eclipses; nodes and apsides of the Moon; Koch and Munkasey
  co-ascendants and the polar ascendant in every house result; sidereal modes 30 to 33 and 39.
- Fixed: `cotrans` and `cotrans_sp` had the opposite sign to the reference; the Midheaven for Regiomontanus, Campanus and
  Topocentric houses; eclipse result shapes, flags and timing entries; star labels and traditional names.

Not built yet: the house systems APC, Sunshine, Savard-A and Pullen sinusoidal ratio; sidereal mode 36; the hypothetical body
Nibiru; a fuller heliacal model. See docs/SWISS_COMPATIBILITY.md.
