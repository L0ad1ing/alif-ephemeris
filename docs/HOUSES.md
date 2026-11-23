# House systems and angles

```python
from alif_ephemeris import houses
h = houses.houses(jd_ut, east_longitude_deg, latitude_deg, "Placidus")
h.cusps        # 12 ecliptic longitudes, house 1 first
h.ascendant, h.midheaven, h.vertex, h.ramc
h.system       # the system actually used (a fallback is reported in h.notes)
```

Systems: Placidus, Koch, Porphyry, Regiomontanus, Campanus, Alcabitius, Topocentric (Polich-Page), Morinus,
Meridian (axial rotation), Carter (poli-equatorial), Equal, Equal (MC), Equal (Aries), Vehlow, Whole sign, Sripati.
Swiss-style one-letter codes (`P K O R C B T M X E W V D N S F`) also work.

Each system is derived from its published geometric definition, using the true obliquity and the local apparent
sidereal time. Measured against Swiss Ephemeris used as a numerical reference, on 60 random places and times
(1900 to 2050, latitudes within 60 degrees): every cusp, the Ascendant, Midheaven and Vertex agree to under
0.007 arcsecond.

## Near the poles

- Placidus and Koch do not exist where a cusp's degree never rises or sets. Like Swiss Ephemeris, we then use
  Porphyry houses and say so in `notes`.
- In 300 random polar cases (latitude 60 to 90, both hemispheres) the Ascendant, Midheaven, Vertex and all
  cusps of Porphyry, Regiomontanus, Campanus, Alcabitius, Equal, Whole sign, Morinus and Meridian agree with
  Swiss Ephemeris. Topocentric differs in about 6% of the cases inside the polar circles, where the system has
  no single sensible answer (cusps 2, 3, 5, 6, 8, 9, 11, 12 can land on the opposite side).

## Added later (2026-11-22)

- **Horizon (azimuthal)**: the horizon is cut into twelve 30 degree arcs from the east point (through the south
  towards the west in the northern hemisphere, the other way in the southern). Each cusp is where the vertical circle
  of a division meets the ecliptic, on the half-circle of that azimuth. The tenth cusp is the Midheaven; the earlier
  attempt failed because this rule was missing. Exact against the reference.
- **Krusinski-Pisa-Goelzer**: the great circle through the Ascendant and the zenith is cut into twelve equal arcs from
  the Ascendant towards the nadir and each division is carried to the ecliptic along its hour circle. Exact.
- **Pullen sinusoidal ratio**: in each quadrant of size Q the two outer houses are Q/4 + 7.5 degrees wide and the
  middle one is Q/2 - 15 degrees. Exact.
- **Gauquelin sectors** (`gauquelin_sectors`): 36 sectors, each an equal share of the diurnal or nocturnal arc of
  its own boundary point (the Placidus idea with 36 parts). Sector 1 starts at the Ascendant.
- **House positions** (`house_pos`): see `house_pos.py`; exact for the systems with a published rule.

## Not included yet

APC, Sunshine (Makransky and Savard), Savard-A and Pullen sinusoidal delta.
