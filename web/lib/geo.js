/*
 * geo.js - direction maths shared by the viewer (index.html) and its tests
 * (tests/test_sun_geo.mjs). Pure functions, no three.js, so Node can test them.
 *
 * World frame used by the viewer:
 *   +X = east, +Y = up, -Z = NORTH.
 * Why -Z: the terrain mesh is a PlaneGeometry rotated -90 deg about X, which
 * puts the texture's top row (image row 0 = north for a north-up GeoTIFF or
 * GAMUS tile) at z = -extent/2. worldToPx() below is the same mapping.
 *
 * Angles follow the map/compass convention:
 *   azimuth = degrees clockwise from north (0 N, 90 E, 180 S, 270 W)
 *   elevation = degrees above the horizon (0 horizon, 90 straight overhead)
 */

const D2R = Math.PI / 180;

/** Unit vector pointing FROM the scene TOWARD the sun. */
export function sunDirection(azDeg, elDeg) {
  const el = Math.max(0, Math.min(90, elDeg)) * D2R;   // below-horizon sun is not lit
  const az = azDeg * D2R;
  const h = Math.cos(el);                                 // horizontal length
  return [h * Math.sin(az), Math.sin(el), -h * Math.cos(az)];
}

/** Horizontal unit direction in which shadows fall (away from the sun), [x, z]. */
export function shadowDirection(azDeg) {
  const az = azDeg * D2R;
  return [-Math.sin(az), Math.cos(az)];
}

/** Compass bearing (deg, 0..360) of a view direction given the fly-mode yaw.
 *  The viewer's forward vector is (sin yaw, ., cos yaw) in (x, z). */
export function viewBearingDeg(yaw) {
  const b = Math.atan2(Math.sin(yaw), -Math.cos(yaw)) / D2R;   // atan2(east, north)
  return (b + 360) % 360;
}

/** Rotation (deg, SVG clockwise-positive) of the compass needle so that it
 *  points at north on screen: straight up when the camera looks north. */
export function compassNeedleDeg(yaw) {
  const r = -viewBearingDeg(yaw);
  return r <= -180 ? r + 360 : r;                       // keep in (-180, 180]
}

/** World (x, z) -> image pixel (column, row) for a W x H grid at gsd metres. */
export function worldToPx(x, z, W, H, gsd) {
  return [(x + (W * gsd) / 2) / gsd, (z + (H * gsd) / 2) / gsd];
}
