# Clean-room rules

Alif Ephemeris must stay free for anyone to use, including in commercial products, with no
licence fee and no obligation to publish their own code. That only holds if nothing in this
repository is derived from code or data that carries a stronger licence.

## Allowed sources
- NASA/JPL ephemeris files (public domain) and the NAIF SPK documentation.
- Published astronomy: Meeus, the Explanatory Supplement to the Astronomical Almanac,
  IAU/IERS conventions, Espenak and Meeus delta-T polynomials (NASA), the Hipparcos catalogue.
- Permissively licensed code (MIT, BSD, Apache) with attribution in NOTICE.
- Classical texts for house systems and ayanamsa definitions, and the published ayanamsa
  reference values.

## Not allowed
- Reading, translating or porting the source code of Swiss Ephemeris (AGPL or professional licence).
- Copying or converting Swiss Ephemeris data files (.se1, sefstars.txt, seas_*.se1, etc.).
- Decompiled or disassembled programs (JHora and others).
- Copying text from the Swiss Ephemeris documentation.

## How Swiss Ephemeris may be used
As a black box, on a developer machine only, to compare numbers (a "test oracle").
Numbers are facts. We never ship it, import it from the package, or vendor its files.
Where Swiss differs from JPL's own data, JPL is the truth and we record the difference.

## Every change
Say in the commit message where a formula or table came from if it is not obvious.

## What we do not ship
The Swiss-style interface does not reproduce the reference program's own small errors. We built tables that could do
that (its sidereal time outside 1850 to 2050, its Uranus, Neptune and Pluto, its mean node, one Pullen house system)
and decided not to ship them: they are fitted to another program's printed numbers, and the free licence is worth
more than matching its mistakes. `docs/SWISS_COMPATIBILITY.md` lists the resulting differences.

## Orbits of the hypothetical bodies
The Uranian planets, Transpluto, Vulcan, Waldemath's dark moon and the other hypothetical bodies are published
astrological conventions. The semi-major axes are the published ones. The remaining elements were recovered by fitting
an orbit to reference positions, and the fitted numbers equal the authors' published elements to the precision of the
fit. They are facts about fictional objects, not code or data from any program.
