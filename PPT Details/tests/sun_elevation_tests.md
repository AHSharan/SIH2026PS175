# Sun direction / elevation and compass: tests

## What the feature is

The 3D viewer (`web/index.html`) has two sliders: **Sun direction** (azimuth,
0–360°) and **Sun elevation** (5–89° above the horizon). They move the
scene's directional light, and with it the shading and cast shadows of
buildings, trees and hills. The compass needle in the bottom bar shows where
north is. This is **lighting and orientation only**: the sun does not enter
the height model or any metric.

The maths lives in `web/lib/geo.js` (pure functions, shared by the viewer and
the tests):

| function | returns |
|---|---|
| `sunDirection(az, el)` | unit vector toward the sun: x = cos(el)·sin(az), y = sin(el), z = −cos(el)·cos(az) |
| `shadowDirection(az)` | horizontal direction shadows fall (away from the sun) |
| `viewBearingDeg(yaw)` / `compassNeedleDeg(yaw)` | compass bearing of the camera, and the needle rotation |
| `worldToPx(x, z, W, H, gsd)` | world metres → image pixel |

World frame: +X east, +Y up, **−Z north**, because the terrain mesh puts the
image's top row (north, for a north-up GeoTIFF or GAMUS tile) at z = −extent/2.

## Bug found while tracing it (fixed)

Before, the viewer computed the sun as `z = +R·cos(el)·cos(az)` and the needle
as `rotate(−yaw)`. Both were mirrored north–south:

| check | before (old code) | after (`geo.js`) | correct |
|---|---|---|---|
| sun at azimuth 0° (north), elevation 0° | (0, 0, **+1**) = south | (0, 0, −1) | north = −Z |
| default sun, azimuth 135° (SE), 42° | (0.525, 0.669, **−0.525**) = north-east | (0.525, 0.669, 0.525) | south-east = +X, +Z |
| compass needle when looking north | **−180°** (points down) | 0° (points up) | 0° |

Commit `b5cec25`. East–west was already right, so the error only showed as
shadows falling the wrong way (north–south) and a needle that pointed south.

## The tests (`tests/test_sun_geo.mjs`, Node's built-in test runner)

| # | test | inputs | expected |
|---|---|---|---|
| 1 | overhead sun | el = 90°, az ∈ {0, 45, 135, 270, 359} | vector = (0, 1, 0) for every azimuth |
| 2 | cardinal azimuths map onto the image | az = 0/90/180/270 at el = 0; 100 × 100 px at 0.5 m | N = −Z and reaches image row 0; E = +X and reaches the last column; S = +Z; W = −X |
| 3 | elevation sets the vertical share | el ∈ {5, 20, 42, 60, 89} × az ∈ {0, 30, 135, 200, 315} | length 1; y = sin(el); horizontal = cos(el) |
| 4 | edge cases | az = 360 vs 0; az = −90 vs 270; el = −10; el = 120 | azimuth wraps; elevation clamped to 0 (horizon) and 90 (overhead) |
| 5 | default light casts sensible shadows | az = 135°, el = 42° | sun is SE (+X, +Z); shadows fall NW, exactly opposite the sun's horizontal direction |
| 6 | compass | yaw for looking N / E / S / W | bearings 0 / 90 / 180 / 270; needle 0 (up) / −90 (left) / ±180 / +90 |
| 7 | the viewer really uses this code | `web/index.html` source | imports `./lib/geo.js`, calls `sunDirection(` and `compassNeedleDeg(yaw)`, and the old formula is gone |

Why these matter: tests 1–4 pin the geometry (unit length, correct split
between height and ground direction, safe inputs). Tests 2, 5 and 6 tie it
to the map: a judge who sets "sun from the south-east" must see shadows fall
north-west, and the needle must point at the image's north. Test 7 stops a
regression where the viewer quietly goes back to its own inline formula.

## Result (run 29 Sep 2026)

```
node --test tests/test_sun_geo.mjs
✔ overhead sun (elevation 90) points straight up for any azimuth
✔ cardinal azimuths land on the right side of a north-up image
✔ elevation sets the vertical share; vector is always unit length
✔ edge cases: azimuth wraps, elevation is clamped to 0..90
✔ viewer default (az 135, SE) casts shadows toward the north-west
✔ compass needle points up when looking north, left when looking east
✔ the viewer really uses these functions (not an old inline formula)
ℹ tests 7   ℹ pass 7   ℹ fail 0
```

Also checked in the running viewer (browser, `window.DW.sun`): slider at 180°
puts the light at z = +520 (south), at 0° at z = −520 (north); the label reads
"135° SE".

## Not covered

- Real sun position for a date/time/place is not computed; the slider is a
  manual light, not an ephemeris.
- Shadow rendering quality (shadow-map resolution) is visual and was checked
  by eye, not by a test.
