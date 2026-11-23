# Plan and status

Goal: every feature of Swiss Ephemeris, plus more, free for anyone to use, including in paid products.
Apache-2.0, credit "Alif Ephemeris by AlifAstrology.com". Publish on GitHub only when the owner says go.

## Done (v0.1, 2026-11-22)
- [x] NASA SPK reader written from the public format notes (agrees with `jplephem` to 1.4 cm).
- [x] IAU 2006/2000A precession and nutation (agrees with SOFA to 0.00003 micro-arcsec).
- [x] Light time, light bending (Sun, Jupiter, Saturn), aberration. Two-part time (no 40 microsecond floor).
- [x] Delta-T: SMH 2016 splines (720 BC to 1972), US Navy measured values and predictions (1973 onward),
      long-term parabola beyond. TDB minus TT.
- [x] Apparent geocentric Sun, Moon, Mercury to Pluto. Tests vs Skyfield and SOFA. Accuracy report.
- [x] Thread-safe, vectorised.

## Done in v0.2 (same day)
- [x] Planet centres for Jupiter to Pluto and NASA's newest Uranus and Neptune orbits, refitted into compact files
      with measured error under 0.01 mas. Matches NASA Horizons to 0.00002 arcsec for every planet.
- [x] Fully vectorised Delta-T, multi-segment batches, one-pass speed, Delta-T error on every result.
- [x] Offline NASA Horizons reference in the tests.

## Done in v0.3 (same day)
- [x] Sidereal time (matches SOFA to under 1 micro-arcsec).
- [x] 15 house systems and the angles (Placidus, Koch, Porphyry, Regiomontanus, Campanus, Alcabitius, Topocentric,
      Morinus, Meridian, Carter, Equal, Equal from MC, Aries, Vehlow, Whole Sign, Sripati), polar fallbacks.
      Matches Swiss to 0.007 arcsec at normal latitudes; polar cases match except Topocentric (see docs/HOUSES.md).
- [x] 36 ayanamsas, sidereal planets, star-based ayanamsas. Lahiri (JHora) = Spica at 180, verified on 518 charts.
- [x] Fixed stars (9,827 Hipparcos stars, 411 IAU names) with proper motion, parallax, aberration, bending.
- [x] Mean and true lunar node, mean and osculating apogee (Lilith).

## Done in v0.4 (same day)
- [x] General positions with observer and correction options (true, J2000, equatorial, heliocentric, topocentric).
- [x] Asteroids and Chiron from NASA Horizons; 16 hypothetical bodies (14 as exact Kepler orbits, 2 circles).
- [x] Lunar, global solar and local solar eclipses (reference calibration: lunar about 1.5 s, local about 1-3 s).
- [x] Swiss-style adapter (`swe_compat`) and a pytest plugin that runs Alif Astrology Software's own tests on it.
- [x] Mean apogee on the inclined orbit (with latitude), matching the reference to 0.5 arcsec.

## Done in v0.5
- [x] Rise, set and transit; planetary phenomena; interpolated apogee; self-hosted web API; several data files together (DE440 + DE441).

## Done in v0.6 (2026-11-22)
- [x] Swiss-style interface completed: house positions, Gauquelin sectors, crossings, refraction, orbital elements, nodes and
      apsides, planetocentric positions, eclipse attributes and visibility, house speeds, true-horizon rise and set.
- [x] House systems: Krusinski, Horizon, Pullen SR and SD, Gauquelin (23 systems).
- [x] Reference conventions in the Swiss-style layer (outer-planet barycentres, mean node, sidereal time).
- [x] Speed: validated Chebyshev windows for planets, lunar points, stars and sidereal time. Alif's own test suite runs
      on the library: 3,100 tests pass; the remaining failures are listed in `docs/SWISS_COMPATIBILITY.md`.
- [x] Packaging: scipy dependency, data files in the wheel, clean-install smoke test.

## Done late on 2026-11-22
- [x] Lunar occultations, house systems Krusinski/Horizon/Pullen/Gauquelin, Vulcan, Waldemath's dark moon, interpolated perigee,
      refrac_extended, orbital elements, memory-mapped DE441, Bayer star codes, star-based sidereal modes as the reference.

## Open items found while testing on Alif
- Global solar eclipse start and end: 5 seconds rms off the reference (the reference's own method is approximate;
  six Alif mundane tests that demand 1.5 s fail).
- Gandhi's Patyayini dasa is 28 minutes off against a 25 minute limit (JHora's own Lagna carries a 0.05 arcsec quirk).
- Not built: APC, Sunshine and Savard-A houses, heliacal events, Nibiru, Saros numbers. Approximate: interpolated perigee.

## Next, in order
1. (done) planet centres.
2. **Our own compact data files** from DE441 (13,200 BC to AD 17,191), with a published error bound per file.
3. **Sidereal zodiacs**: all ayanamsas (Lahiri, Raman, Krishnamurti, Fagan-Bradley, true Chitra, Yukteshwar,
   JN Bhasin, the galactic ones and more), tested against JHora and published reference values.
4. **Houses**: Placidus, Koch, Porphyry, Regiomontanus, Campanus, Equal, Whole Sign, Alcabitius, Morinus,
   Topocentric, Meridian, Vehlow, Sripati, Krusinski and the polar fallbacks. Sidereal time, MC, Ascendant, Vertex.
5. **Points**: mean and true lunar node, mean and osculating Lilith, interpolated apogee and perigee, Chiron,
   Pholus, Ceres, Pallas, Juno, Vesta and the other bodies already used in Alif (42 asteroids from JPL).
6. **Fixed stars** from the Hipparcos catalogue with proper motion. Planetary and star magnitudes.
7. **Events**: sunrise, sunset, moonrise, twilight, eclipses (solar and lunar), conjunction and ingress finding,
   heliacal risings, tithi and nakshatra helpers.
8. **Switch Alif over**: desktop app, website engine, iPhone app. Remove Swiss Ephemeris from all three.
9. **TypeScript port** from the same test vectors (website and iPhone).
10. **Public release**: README polish, examples, docs site, PyPI and npm, GitHub (needs owner go).

## What "better than Swiss Ephemeris" means here (measurable)
- Free for commercial use with no source disclosure (done).
- Delta-T uses current measured values (done; Swiss build tested was 0.3 s low for 2026).
- Same or better planet accuracy (done: matches NASA Horizons to 0.00002 arcsec for every planet; Swiss is up to 0.24 arcsec off for Uranus).
- Thread-safe, no global state, no file-handle limits (done).
- Error bounds returned with results (Delta-T effect done; ephemeris bounds in step 2).
- Public test suite and accuracy report anyone can run (done, grows each step).
- Same features as Swiss, then more (steps 3 to 7).

## Open decisions for the owner
- When to make the repo public.
- Whether to register the package name `alif-ephemeris` on PyPI and npm.
- Whether to add a short "Powered by Alif Ephemeris" line inside Alif apps (suggested, not required).
