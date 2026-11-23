# Alif Ephemeris

A free, open-source ephemeris for astrology and astronomy software, by [AlifAstrology.com](https://alifastrology.com).

- Free for everyone, including commercial use. No licence fee, no need to publish your own code.
- Apache-2.0 licence. The only condition is that you keep the `NOTICE` file with your copy.
  If you use Alif Ephemeris, please credit **"Alif Ephemeris by AlifAstrology.com"**.
- Built on NASA/JPL's public-domain planetary data.
- Help, questions, bug reports: **info@alifastrology.com**

> Status: version 1.0. Everything Swiss Ephemeris is used for in astrology software works and is tested: planets,
> houses (23 systems), ayanamsas, fixed stars, lunar points, asteroids, hypothetical bodies, eclipses, rise and set,
> phenomena, orbital elements, crossings and more. Software written for the Swiss interface runs on it
> (`docs/SWISS_COMPATIBILITY.md`), and you can run your own web API (`docs/API.md`). `docs/PLAN.md` lists what is next.

## What works today

- Apparent geocentric positions of the Sun, Moon, Mercury to Pluto. Longitude, latitude, distance and daily speed,
  on the true ecliptic and equinox of date. Astrometric ICRF positions too.
- True planet centres for Jupiter to Pluto (not system barycentres), and NASA's newest orbit solutions for
  Uranus and Neptune.
- Full corrections: light travel time, light bending by the Sun, Jupiter and Saturn, aberration,
  IAU 2006 precession and IAU 2000A nutation.
- Time: Julian Day, Delta-T from 720 BC onward (measured values to October 2026, US Navy predictions after),
  TT and TDB. Every result reports how much Delta-T uncertainty could move it (`delta_t_error_arcsec`).
- Sidereal time, the angles (Ascendant, Midheaven, Vertex) and 23 house systems (`docs/HOUSES.md`), Gauquelin sectors and
  the house position of any point.
- 36 ayanamsas and sidereal positions (`docs/AYANAMSAS.md`). "Lahiri (JHora)" is Spica at 180 degrees and matches
  518 JHora charts to 0.02 arcsecond.
- 9,827 fixed stars (Hipparcos) with 411 official IAU names, proper motion and parallax.
- Mean and true lunar node, mean and osculating lunar apogee (Black Moon Lilith).
- Chiron, Pholus, Ceres, Pallas, Juno and Vesta from NASA's current orbit solutions (1800 to 2200).
- 18 hypothetical bodies: the Uranian planets, Transpluto, Harrington's Planet X, the old Neptune and Pluto
  predictions, White Moon (Selena), Proserpina, Vulcan and Waldemath's dark moon, as orbits.
- Observer and correction choices: true or apparent positions, J2000 or of date, with or without nutation,
  equatorial or ecliptic, heliocentric, barycentric, or topocentric for a place on Earth.
- Lunar eclipses, solar eclipses (whole Earth and for one place) with type, contact times, magnitudes, visibility
  from a place and where the shadow falls.
- Osculating orbital elements, nodes and apsides of the planets, planetocentric positions, and the next time the Sun,
  Moon or a planet reaches a longitude.
- Rising, setting and meridian transit times of the Sun, Moon, planets and stars (a few seconds from the reference).
- Phase, illuminated fraction, elongation, apparent size and magnitude of the planets (Mallama and Hilton 2018).
- A Swiss-style function interface, so existing software can run on it unchanged, and a self-hosted web API.
- Thread-safe, no global state. Any number of threads can share one `Ephemeris`.
- Speed: in the Swiss-style interface a planet position with speed takes about 6 microseconds once the fast windows
  are built, the same as Swiss Ephemeris. Houses, azimuth and altitude, sidereal time, fixed stars and the ayanamsa are
  within a few microseconds of it, and a rise or set takes about 150 microseconds. A batch of full apparent positions runs
  at about 19,000 per second. `python tools/bench_speed.py` prints the comparison on your machine.
- Dates from year -1000 to 3200: DE440 covers 1550 to 2650; two small range packs cut straight from DE441
  (`tools/make_range_pack.py`, about 340 MB together) cover the rest with identical numbers. Without them the DE441 files
  work as before.

## Quick start

```python
from alif_ephemeris import Ephemeris, MOON, julian_day

eph = Ephemeris("data/raw/de440s.bsp")          # see docs/DATA.md for the free download
jd = julian_day(2026, 11, 22, 12.0)             # 22 Nov 2026, 12:00 UT

moon = eph.apparent(MOON, jd_ut=jd)
print(moon.longitude, moon.latitude, moon.longitude_speed, moon.delta_t_error_arcsec)
```

Install for development: `pip install -e ".[dev]"` then `python -m pytest` (tests that need the NASA file skip if it is missing).

## Using it instead of Swiss Ephemeris

Alif Ephemeris was built so you do not need to pay for, or open-source your code because of, the Swiss Ephemeris.

1. Install it: `pip install git+https://github.com/L0ad1ing/alif-ephemeris` (it needs `numpy` and `scipy`).
2. Download NASA's free planet file once (`de440.bsp`, 114 MB, see `docs/DATA.md`) and point to it:
   `ALIF_EPHEMERIS_FILE=/path/to/de440.bsp`.
3. Software written for the Swiss `swisseph` module runs unchanged. Put our adapter under that name before the
   software imports it:

```python
import sys
import alif_ephemeris.swe_compat as swe
sys.modules["swisseph"] = swe

print(swe.calc_ut(swe.julday(2026, 11, 22, 12.0), swe.SUN, swe.FLG_SWIEPH | swe.FLG_SPEED)[0])
print(swe.houses_ex(swe.julday(2026, 11, 22, 12.0), 39.0, -77.0, b"P")[1][:2])      # Ascendant, Midheaven
```

The adapter has about 150 Swiss-style functions (positions, houses, ayanamsas, fixed stars, eclipses, occultations,
rise and set, crossings, orbital elements and more). `docs/SWISS_COMPATIBILITY.md` lists what is covered, what is
not, and the small differences you will see. For new code use the main `Ephemeris` class, which is faster for many
dates at once and reports its own error bounds.

## How accurate is it?

We compare ourselves with **NASA JPL Horizons**, JPL's official service, on 20 random dates from 1950 to 2050
(`python tools/horizons_check.py`). Astrometric ICRF place, worst angular difference in arcseconds.
The Swiss Ephemeris column is the Swiss Ephemeris 2.10.03 build and the data files on the machine we tested on
(ICRS, no aberration or light bending, same Terrestrial Time).

| Body | Alif Ephemeris | Swiss Ephemeris |
|---|---|---|
| Sun | 0.00000 | 0.0017 |
| Moon | 0.0025 (see note) | 0.0033 |
| Mercury | 0.00000 | 0.0017 |
| Venus | 0.00000 | 0.0013 |
| Mars | 0.00000 | 0.0034 |
| Jupiter | 0.00001 | 0.052 |
| Saturn | 0.00000 | 0.043 |
| Uranus | 0.00002 | 0.24 |
| Neptune | 0.00001 | 0.12 |
| Pluto | 0.00001 | 0.14 |

For the Moon, Horizons uses the DE441 ephemeris and we use DE440, which JPL recommends for 1550 to 2650.
They agree to about a metre near 1969 and drift apart slowly (0.001 arcsec in 2030, 0.008 in 2145).

The full "apparent of date" pipeline (light time, bending, aberration, precession, nutation) agrees with
Skyfield and the IAU SOFA routines to better than 0.00001 arcsec (`tests/test_core_vs_oracles.py`).

Delta-T (the gap between clock time and Earth-rotation time) matters more than any of this.
For 2026 the measured value is 69.17 s. The Swiss Ephemeris build we tested uses 68.85 s, which moves the
Moon by about 0.17 arcsec. Our values follow the US Navy's measurements and predictions.

## Limits you should know

- The small NASA file we start with (DE440s) covers 1849 to 2150. Bigger coverage comes with our own
  data files (`docs/PLAN.md`).
- Planet-centre offsets cover 1800 to 2200 and the newer Uranus and Neptune orbits cover 1850 to 2150.
  Outside those dates the system barycentre and DE440's orbits are used (up to about 0.1 to 0.35 arcsec off for
  the outer planets).
- Mars is the Mars-system barycentre, which is within about 0.2 m of the planet centre.
- Delta-T for the future is a prediction. It is reported, not hidden: see `delta_t_uncertainty`.
- Our refitted data files carry measured errors under 0.01 milli-arcsecond against NASA's own satellite data.
- Mean Lilith is the mean lunar apogee on the Moon's inclined mean orbit, with its ecliptic latitude (up to 5.1 degrees);
  it matches the reference to 0.5 arcsecond.
- Topocentric houses differ from Swiss in about 6% of cases inside the polar circles (no single right answer there).
- Eclipse start and end for the whole Earth agree with the reference to about 5 seconds (up to 13 for partial
  eclipses); lunar and local solar eclipse times to a few seconds.
- Dates: DE440 covers 1550 to 2650. With the two DE441 files (`docs/DATA.md`) the range is about 13,000 BC to AD 17,000,
  with the usual loss of accuracy for the planet-centre tables outside 1800 to 2200.
- Not built yet: the APC, Sunshine and Savard-A house systems, heliacal events, Nibiru, Saros numbers.
- Approximate: the interpolated perigee (median 0.3 degrees from the reference, up to 6), the interpolated apogee
  (within 0.2), Waldemath's dark moon (about 1 arcminute) and Vulcan (a few arcseconds).
- The Swiss-style layer follows the reference program's conventions by default (planet-system barycentres for the
  outer planets, its sidereal-time convention outside 1850 to 2050, its mean node). `set_reference_convention(False)`
  switches to the exact values; the main `Ephemeris` class always gives the exact values.

## How we keep this free

This is a clean-room project. We never read, copy or port Swiss Ephemeris code or data. Swiss Ephemeris
is used only to compare numbers on a developer machine. See `docs/CLEAN_ROOM.md`.

Thank you to NASA/JPL, the US Naval Observatory, the IAU and IERS, and the authors of the published
models listed in `NOTICE`.
