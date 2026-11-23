# Running your own web API

Alif Ephemeris includes a small web server you start on your own computer. It uses only Python's standard library,
listens on `127.0.0.1` (your machine only) unless you choose otherwise, costs nothing, and sends nothing anywhere.

```
python -m alif_ephemeris.server --file data/raw/de440.bsp --port 8765
```

Then open these in a browser or call them from any language:

| Route | What it returns | Example |
|---|---|---|
| `/api/health` | coverage of the data file, counts | `/api/health` |
| `/api/position` | longitude, latitude, distance, speed of a body (add `sidereal=Lahiri (JHora)`) | `/api/position?body=moon&jd=2461000.5` |
| `/api/houses` | cusps and angles | `/api/houses?jd=2461000.5&lon=-77&lat=39.4&system=P` |
| `/api/ayanamsa` | a value, or the list of names | `/api/ayanamsa?name=Lahiri%20(JHora)&jd=2461000.5` |
| `/api/star` | a fixed star's place | `/api/star?name=Spica&jd=2461000.5` |
| `/api/riseset` | next rise, set and upper transit | `/api/riseset?body=sun&jd=2461000.5&lon=-77&lat=39.4` |
| `/api/eclipse` | next (or `back=1` previous) lunar or solar eclipse | `/api/eclipse?kind=lunar&jd=2461000.5` |

Give the time as `jd=` (Julian Day, UT) or `date=2026-11-22T12:00:00` (UT).
Errors come back as JSON with a plain message and an HTTP 400.

Each request takes a few milliseconds. This server is meant for your own use or a small trusted group. If you want a
public one, put it behind your own rate limiting; the project does not run one.
