# Ayanamsas (sidereal zodiacs)

An ayanamsa is the angle between the moving tropical zero point (the vernal equinox) and the sidereal zero
point (fixed against the stars). Sidereal longitude = apparent longitude minus the **true** ayanamsa
(mean ayanamsa plus nutation in longitude).

Use it from Python:

```python
from alif_ephemeris import Ephemeris, ayanamsa
ayanamsa.names()                              # every system
ayanamsa.ayanamsa("Lahiri (JHora)", jd_tt)    # degrees, mean (add true=True for nutation)
eph.apparent(MOON, jd_ut=jd, sidereal="Lahiri (JHora)")
```

## How each kind is defined

**Star based** (exact at every date, including the far past and future, with proper motion):
the star or point is held at a fixed sidereal longitude.

| System | Held at |
|---|---|
| Lahiri (JHora) and True Chitra (Spica 180) | Spica at 180 degrees |
| True Revati | zeta Piscium at 359 deg 50 min |
| True Pushya | delta Cancri at 106 degrees |
| True Mula | lambda Scorpii at 240 degrees |
| Galactic Centre 0 Sagittarius / Cochrane | Sgr A* (ICRS 266.416837, -29.007810) at 240 / 270 degrees |

**Lahiri (JHora) is Spica at 180 degrees.** We checked this against 540 JHora charts from 1835 to 2010: 518 use Lahiri and agree with Spica-at-180 to
better than 0.02 arcsecond (worst 0.017), which is the precision JHora prints; the other 22 use different ayanamsas
(`tools/check_star_ayanamsa.py`).
Swiss Ephemeris calls a slightly different value "Lahiri" (about 56 arcsec larger at 2000), which is why this
library offers both: **Lahiri** (the IAE-style value, 23.857092 degrees at J2000) and **Lahiri (JHora)**
(23.841360 degrees at J2000). Pick the one your reference programs use.

**Fixed point** (Fagan-Bradley, Raman, Krishnamurti, Yukteshwar, DeLuce, Babylonian, Suryasiddhanta,
Aryabhata, and the rest): the value at J2000.0 is the numerical definition of the convention. It is carried to
other dates by exact precession of that point (IAU 2006). Against the reference these agree to 0.01 arcsecond
for most systems and to 0.03 to 0.09 arcsecond for the ancient-epoch ones (Babylonian, Sassanian,
Suryasiddhanta, DeLuce) between 1800 and 2200. Over several thousand years the IAU 2006 polynomial
accumulates a few arcseconds of difference; a long-term precession model is on the plan.

## Not included yet

True Sheoran, Galactic equator systems (IAU 1958, true, Mula), Gil Brand and Wilhelm galactic centre
variants, Hipparchos star-based variant, and user-defined ayanamsas. The star-based ones in the table above
differ from Swiss Ephemeris's versions by 3 to 20 arcseconds because the conventions place the star or
galactic centre slightly differently; ours are explicit and documented.
