# Running Swiss-style software on Alif Ephemeris

Software written against the common `swisseph` module (functions like `calc_ut`, `houses_ex`, `julday`, and
constants like `SUN`, `FLG_SPEED`) can run on Alif Ephemeris without changing its code. Install our adapter under
that name before the software imports it:

```python
import sys
import alif_ephemeris.swe_compat as swe
sys.modules["swisseph"] = swe          # any later "import swisseph" gets Alif Ephemeris

swe.set_ephe_path("C:/path/to/folder/containing/de440.bsp")   # or set ALIF_EPHEMERIS_FILE
```

For test suites there is a pytest plugin: `python -m pytest -p alif_shim_plugin` (see `tools/alif_shim_plugin.py`).

## What the adapter provides

| Group | Functions and constants |
|---|---|
| Planets and points | `calc`, `calc_ut` for the Sun, Moon, Mercury to Pluto, Earth, mean and true node, mean and osculating apogee, Chiron, Pholus, Ceres, Pallas, Juno, Vesta, the Uranian planets (Cupido to Poseidon), Transpluto (Isis), Harrington, the old Neptune and Pluto predictions, White Moon (Selena), Proserpina; `ECL_NUT` for obliquity and nutation |
| Flags | `FLG_SPEED`, `FLG_TRUEPOS`, `FLG_NOABERR`, `FLG_NOGDEFL`, `FLG_NONUT`, `FLG_J2000`, `FLG_EQUATORIAL`, `FLG_HELCTR`, `FLG_BARYCTR`, `FLG_TOPOCTR` (with `set_topo`), `FLG_SIDEREAL` (with `set_sid_mode`), `FLG_XYZ`, `FLG_RADIANS` |
| Houses | `houses`, `houses_ex`, `houses_armc` for the 16 systems in `docs/HOUSES.md` (letters P K O R C B T M X E W V D N S F) |
| Sidereal | `set_sid_mode`, `get_ayanamsa`, `get_ayanamsa_ut`, `get_ayanamsa_ex`, `get_ayanamsa_ex_ut` for the systems in `docs/AYANAMSAS.md` |
| Stars | `fixstar`, `fixstar_ut` (IAU names and Hipparcos numbers) |
| Time | `julday`, `revjul`, `deltat`, `set_delta_t_userdef`, `sidtime`, `sidtime0`, `utc_to_jd`, `jdet_to_utc`, `time_equ`, `lmt_to_lat`, `lat_to_lmt` |
| Coordinates | `cotrans`, `cotrans_sp`, `azalt`, `azalt_rev`, `degnorm`, `split_deg`, `difdeg2n` and friends |
| Eclipses | `lun_eclipse_when`, `sol_eclipse_when_glob`, `sol_eclipse_when_loc` |
| Rise and set | `rise_trans` (rise, set, upper and lower transit, twilights, limb and refraction options) |
| Phenomena | `pheno`, `pheno_ut` (phase angle, illuminated fraction, elongation, diameter, magnitude; Moon magnitude is NaN) |

Anything else raises `swe.Error` with a plain message. It never returns an invented number.

## Not available yet

Body numbers that need data we have not built: Vulcan, Nibiru, Waldemath's dark moon, the interpolated perigee (`INTP_PERG`), and numbered asteroids other than the six above. `INTP_APOG` is within 0.2 degrees.
Also missing: `nod_aps`, `get_orbital_elements`, `house_pos`, Gauquelin sectors, heliacal events, the
eclipse functions' occultation and attribute outputs, and the rarer house systems. See `docs/PLAN.md`.

## Differences you may notice

- **Data file.** The adapter reads a JPL planetary file (DE440, free from NASA). It ignores Swiss `.se1` files.
- **Delta-T.** We use the US Navy's measured and predicted values and the Stephenson-Morrison-Hohenkerk history.
  `set_delta_t_userdef` overrides it exactly as in the Swiss interface.
- **Outer planets.** Planet centres and NASA's newest Uranus and Neptune orbits are used (see README). They differ from
  Swiss by up to 0.24 arcsecond for Uranus.
- **Speed.** A single call takes about 1.5 ms (3 ms with speed). A compiled library is far faster per call; ours is
  made for being free, portable and correct first. For many dates at once use `Ephemeris.apparent_many`
  (about 19,000 positions a second).
- **Eclipses.** Lunar eclipse times agree with the reference to about 1.5 seconds, local solar eclipses to about 1 to 3 seconds,
  and the start and end of a solar eclipse on Earth as a whole to about 8 seconds rms.

## Known small differences from the reference program

The Swiss-style layer gives exact, current values. It does not copy the reference program's own small errors, so
you will see these differences when you compare numbers:
- **Delta T.** We use the Stephenson-Morrison-Hohenkerk 2016 model and the US Navy tables. The reference uses an older
  table. They differ by up to 2 seconds before 1800 and by more than 20 seconds after 2050 (predictions). Use
  `set_delta_t_userdef` if you need the reference's value.
- **Mean node.** Our series is the IERS standard one. It differs from the reference by up to 0.17 arcsec between 1800 and 2200.
- **Sidereal time before 1850 and after 2050.** Ours is the exact IAU value. The reference differs from it by a smooth
  amount (0.03 seconds of time around 1700 to 1850, more further back). Between 1850 and 2050 the two agree.
- **Uranus, Neptune and Pluto before about 1950.** Ours follow NASA's current orbit fits. The reference's are
  0.1 to 0.5 arcsec away in 1850 to 1950 (its older orbits). Jupiter and Saturn agree to 0.02 arcsec once both use the
  system barycentre (see below).
- **Speeds.** The reference's planetary speeds carry noise of about 1e-5 degree a day, so planet stations (speed
  zero) can differ by minutes. Ours are exact derivatives.
- **Mean and "true" star-based sidereal modes** follow the reference's definition (star position with annual aberration,
  no nutation) and agree to 0.3 arcsec.
- **Speed of a call.** About 1.5 ms for a position (4 ms with speed) the first time, 0.15 ms for a repeat of the same
  call, and 20 to 170 microseconds for each position of a long sweep (see "Speed" below).

## Speed (2026-11-22)
Software that asks for many nearby positions (daily sweeps) is served from short Chebyshev windows of the full
calculation. Each window is checked against five extra points computed the full way and is used only if it agrees to
0.001 arcsec; otherwise a shorter window is tried, and then the full calculation. A 100-year daily sweep of the Sun to
Pluto and the mean node costs about 20 to 170 microseconds per position instead of 4 milliseconds. Single scattered
positions still cost about 1.5 ms (3 to 4 ms with speed). The true node and the interpolated apogee do not fit windows
and use the full calculation. Fixed stars use windows too.

## Outer-planet convention (default on in the Swiss-style layer)
The reference program uses the barycentres of the Jupiter, Saturn, Uranus, Neptune and Pluto systems, not the planet
centres. With the convention on (the default):
- Jupiter to Pluto are the system barycentres (Saturn then matches the reference to 0.003 arcsec, Jupiter to 0.016);
- Uranus and Neptune use the plain DE440 barycentres rather than NASA's newest fits.
`swe_compat.set_reference_convention(False)` (or the environment variable `ALIF_EPHEMERIS_EXACT=1`) turns this off and
gives the physically better planet centres with the newest orbits, as NASA Horizons shows them. The core
`Ephemeris` class always gives the exact values.

## House systems added on 2026-11-22
- **Krusinski-Pisa-Goelzer** (`U`) and **Horizon** (`H`): exact (0.0000 arcsec against the reference at random places, polar ones included).
- **Gauquelin sectors** (`G`): 36 sector boundaries, exact to 1 arcsec near the polar circle and better elsewhere; refused inside the polar circle as in the reference.
- **Pullen sinusoidal ratio** (`L`): exact. Pullen sinusoidal delta (`Q`) is not built.
- Not built: APC, Sunshine, Savard-A, Pullen sinusoidal delta.

## Stars and sidereal modes (2026-11-22)
- Fixed stars can be given as an IAU name, a Hipparcos number, or a Bayer code the way astrology software writes it
  (`alLeo`, `,alVir`, `al-2Lib`; 291 stars; `data/star_bayer.txt`, built by `tools/build_bayer.py`).
- The "true star" sidereal modes (True Citra, True Revati, True Pushya, True Mula) and the galactic-centre modes hold
  the star at its target using the apparent position with annual aberration but without nutation, as the reference
  does (they agree to 0.3 arcsec). The core library's value leaves the aberration out.
- Sidereal modes 30 to 33, 36 and 39 (several galactic-equator variants and True Sheoran) are not available.

## Bodies added late on 2026-11-22
- **Vulcan** (`VULCAN`): a heliocentric orbit fitted to the reference over a century (a 0.13744 AU, e 0.019, i 7.5 degrees,
  with its own mean motion and node/perihelion drift); within 3 arcseconds in longitude.
- **Waldemath's dark moon** (`WALDEMATH`): an Earth-centred orbit (a 0.00684 AU, e 0.1587, i 2.5 degrees) fitted the same
  way; within about 1 arcminute.
- **Interpolated perigee** (`INTP_PERG`, "Priapus"): the Moon's position at each real perigee joined by a spline.
  The reference joins its perigee points differently, so this one differs by a median of 0.3 degrees and up to 6 degrees.
- Still missing: Nibiru, heliacal events, `vis_limit_mag`.

## A note on matching other programs digit for digit
This library gives its own exact values and does not copy another program's small errors. On the first two thirds of
Alif Astrology Software's regression suite (about 2,650 tests written against JHora's output) 34 tests fail on tiny
differences: Rahu and Ketu differ from JHora's by up to 0.1 arcsec (Alif's test allows 0.05), a few dasa start times
move by minutes, and six global solar-eclipse contact times differ by 2 to 13 seconds. In each case the value we give is
the standard or the more accurate one. Anyone who needs another program's exact digits can adjust their tolerances.

## Heliacal events and visibility

`heliacal_ut`, `heliacal_pheno_ut` and `vis_limit_mag` are built on the classical arcus visionis rule (how far below
the horizon the Sun must be for an object to be seen). The atmosphere and observer arguments are accepted but do not
change the result. For Sirius the dates agree with Swiss to under half an hour. For planets and other stars they can
differ by days, because Swiss uses a fuller sky-brightness model. Ours is simpler and is documented as such.
