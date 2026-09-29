// Tests for web/lib/geo.js: sun direction, shadows, compass, pixel mapping.
// Run:  node --test tests/test_sun_geo.mjs
// Documented in "PPT Details/tests/sun_elevation_tests.md".
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { sunDirection, shadowDirection, viewBearingDeg, compassNeedleDeg, worldToPx }
  from '../web/lib/geo.js';

const close = (a, b, eps = 1e-9, msg = '') =>
  assert.ok(Math.abs(a - b) < eps, `${msg} expected ${b}, got ${a}`);
const len = v => Math.hypot(...v);

test('overhead sun (elevation 90) points straight up for any azimuth', () => {
  for (const az of [0, 45, 135, 270, 359]) {
    const [x, y, z] = sunDirection(az, 90);
    close(y, 1, 1e-12, `az ${az} y`); close(x, 0, 1e-12, 'x'); close(z, 0, 1e-12, 'z');
  }
});

test('cardinal azimuths land on the right side of a north-up image', () => {
  // world: +X east, -Z north (image row 0)
  const n = sunDirection(0, 0), e = sunDirection(90, 0), s = sunDirection(180, 0),
        w = sunDirection(270, 0);
  close(n[2], -1, 1e-12, 'north z'); close(e[0], 1, 1e-12, 'east x');
  close(s[2], 1, 1e-12, 'south z'); close(w[0], -1, 1e-12, 'west x');
  // and in pixels: moving toward a north sun from the centre reaches row 0,
  // toward an east sun reaches the last column (100 x 100 px at 0.5 m)
  const [, rowN] = worldToPx(n[0] * 25, n[2] * 25, 100, 100, 0.5);
  const [colE] = worldToPx(e[0] * 25, e[2] * 25, 100, 100, 0.5);
  close(rowN, 0, 1e-9, 'north sun -> image row'); close(colE, 100, 1e-9, 'east sun -> image column');
});

test('elevation sets the vertical share; vector is always unit length', () => {
  for (const el of [5, 20, 42, 60, 89]) for (const az of [0, 30, 135, 200, 315]) {
    const v = sunDirection(az, el);
    close(len(v), 1, 1e-12, `|v| az ${az} el ${el}`);
    close(v[1], Math.sin(el * Math.PI / 180), 1e-12, 'y = sin(el)');
    close(Math.hypot(v[0], v[2]), Math.cos(el * Math.PI / 180), 1e-12, 'horizontal = cos(el)');
  }
});

test('edge cases: azimuth wraps, elevation is clamped to 0..90', () => {
  const same = (a, b) => a.forEach((v, i) => close(v, b[i], 1e-12, `component ${i}`));
  same(sunDirection(360, 30), sunDirection(0, 30));
  same(sunDirection(-90, 30), sunDirection(270, 30));
  close(sunDirection(135, -10)[1], 0, 1e-12, 'below horizon -> horizon');
  close(sunDirection(135, 120)[1], 1, 1e-12, 'over 90 -> overhead');
});

test('viewer default (az 135, SE) casts shadows toward the north-west', () => {
  const [sx, , sz] = sunDirection(135, 42);
  assert.ok(sx > 0 && sz > 0, 'sun is south-east (+X, +Z)');
  const [dx, dz] = shadowDirection(135);
  assert.ok(dx < 0 && dz < 0, 'shadow falls north-west (-X, -Z)');
  // a shadow points exactly away from the sun's horizontal direction
  const h = Math.hypot(sx, sz);
  close(dx, -sx / h, 1e-12); close(dz, -sz / h, 1e-12);
});

test('compass needle points up when looking north, left when looking east', () => {
  // fly-mode forward vector is (sin yaw, cos yaw) in (x, z)
  const yawN = Math.PI, yawE = Math.PI / 2, yawS = 0, yawW = -Math.PI / 2;
  close(viewBearingDeg(yawN), 0, 1e-9, 'bearing N'); close(viewBearingDeg(yawE), 90, 1e-9, 'bearing E');
  close(viewBearingDeg(yawS), 180, 1e-9, 'bearing S'); close(viewBearingDeg(yawW), 270, 1e-9, 'bearing W');
  close(compassNeedleDeg(yawN), 0, 1e-9, 'needle N');
  close(compassNeedleDeg(yawE), -90, 1e-9, 'needle E');       // north is to the left
  close(Math.abs(compassNeedleDeg(yawS)), 180, 1e-9, 'needle S');
  close(compassNeedleDeg(yawW), 90, 1e-9, 'needle W');
});

test('the viewer really uses these functions (not an old inline formula)', () => {
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  assert.ok(html.includes("from './lib/geo.js'"), 'index.html imports lib/geo.js');
  assert.ok(html.includes("sunDirection(+$('saz').value"), 'sun position comes from sunDirection()');
  assert.ok(html.includes('compassNeedleDeg(yaw)'), 'needle comes from compassNeedleDeg()');
  assert.ok(!html.includes('R*Math.cos(el)*Math.cos(az))'), 'old formula (az 0 = south) is gone');
});
