# Data sources

None of these files are committed (except the small ones in `src/alif_ephemeris/data/`).

| File | Where from | Size | Licence |
|---|---|---|---|
| `data/raw/de440s.bsp` | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp | 32 MB | NASA/JPL, public domain |
| `src/alif_ephemeris/data/deltat_observed_usno.txt` | https://maia.usno.navy.mil/ser7/deltat.data | 14 KB | US Government, public domain |
| `src/alif_ephemeris/data/deltat_predicted_usno.txt` | https://maia.usno.navy.mil/ser7/deltat.preds | 2 KB | US Government, public domain |
| `src/alif_ephemeris/data/deltat_smh_splines.npz` | Stephenson, Morrison, Hohenkerk (2016), table S15, 2020 update | 5 KB | Published model parameters |
| `src/alif_ephemeris/data/planet_centres.npz` | Refit of NASA satellite kernels (jup365, sat441, ura184, nep098, plu060) by `tools/build_planet_centres.py` | 17 MB | NASA/JPL, public domain |
| `src/alif_ephemeris/data/stars.npz` | ESA Hipparcos (via NASA HEASARC `hipparcos` table) + IAU-CSN star names, by `tools/build_stars.py` | 260 KB | ESA public data; IAU names CC BY |
| `src/alif_ephemeris/data/equinox_terms.npz` | IERS Conventions 2010 table 5.2e, by `tools/export_ee_tables.py` | 3 KB | International standard |
| `src/alif_ephemeris/data/smallbodies.npz` | NASA JPL Horizons orbit solutions for 6 bodies, `tools/build_smallbodies.py` | 10 MB | NASA/JPL, public domain |
| `src/alif_ephemeris/data/star_bayer.txt` | Bayer codes of the IAU-named stars, by `tools/build_bayer.py` (from IAU-CSN) | 4 KB | IAU names CC BY |
| `data/raw/de441_part-1.bsp`, `de441_part-2.bsp` | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/ (optional, about 13,000 BC to AD 17,000) | 1.6 GB each | NASA/JPL, public domain |
| `data/raw/de440.bsp` | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp (1550 to 2650) | 114 MB | NASA/JPL, public domain |
| `tests/data/eclipse_reference.json` | Reference eclipse circumstances 1950 to 2020 | 5 KB | Numbers only |
| `tests/data/horizons_reference.json` | NASA JPL Horizons answers (`tools/make_horizons_fixture.py`) | 12 KB | NASA/JPL, public domain |
| `src/alif_ephemeris/data/iau2000a.npz` | IAU 2000A nutation series (IERS Conventions) | 13 KB | International standard |

Refresh the two US Navy files every few months to keep Delta-T current:
`curl -o src/alif_ephemeris/data/deltat_observed_usno.txt https://maia.usno.navy.mil/ser7/deltat.data` and the same for `deltat.preds`.

Rebuild `planet_centres.npz` with `PYTHONPATH=src python tools/build_planet_centres.py` (needs network; it reads
only the records it needs, about 100 MB, from NASA's servers).

Optional, for dates before 1550 and after 2650 (to about 13,000 BC and AD 17,000): the two DE441 files, 1.65 GB each
(`de441_part-1.bsp`, `de441_part-2.bsp` from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/ ). Put them
next to `de440.bsp`; the library uses DE440 where it applies and DE441 elsewhere. NASA/JPL, public domain.
Planet-centre and Uranus/Neptune corrections cover 1800 to 2200; asteroids 1800 to 2200.
