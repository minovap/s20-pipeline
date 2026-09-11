// Per-cloud orientation calibration: a rotation (roll, pitch, yaw in degrees,
// applied as yaw·pitch·roll about the cloud's preview origin) and a
// translation in metres. Used identically by the renderer, slice masks and
// the Python exporter.
import * as THREE from 'three';
import type {Orientation} from '../types';

export const IDENTITY: Orientation = {rotation: [0, 0, 0], translation: [0, 0, 0]};
const deg = Math.PI / 180;

/** Row-major 3x3 rotation matrix for an orientation. */
export function rotationMatrix(o: Orientation): number[] {
  const e = new THREE.Euler(o.rotation[0] * deg, o.rotation[1] * deg, o.rotation[2] * deg, 'ZYX');
  const m = new THREE.Matrix4().makeRotationFromEuler(e).elements; // column-major
  return [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]];
}
export function quaternion(o: Orientation): THREE.Quaternion {
  return new THREE.Quaternion().setFromEuler(new THREE.Euler(o.rotation[0] * deg, o.rotation[1] * deg, o.rotation[2] * deg, 'ZYX'));
}
export const isIdentity = (o: Orientation | undefined) => !o || (o.rotation.every(v => v === 0) && o.translation.every(v => v === 0));

/** Compose a rotation about a world axis (through the cloud origin) onto an orientation. */
export function rotateAboutAxis(o: Orientation, axis: [number, number, number], degrees: number): Orientation {
  const dq = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(...axis).normalize(), degrees * deg);
  const total = dq.clone().multiply(quaternion(o));
  const e = new THREE.Euler().setFromQuaternion(total, 'ZYX');
  const t = new THREE.Vector3(...o.translation).applyQuaternion(dq);
  const r = (v: number) => Math.round(v * 1000) / 1000;
  return {rotation: [r(e.x / deg), r(e.y / deg), r(e.z / deg)], translation: [r(t.x), r(t.y), r(t.z)]};
}

/** Transform passed to unionMask and the exporter: world = R·local + origin + t. */
export type Transform = {rotation: number[]; origin: number[]; translation: number[]};
export function transformFor(o: Orientation | undefined, origin: number[]): Transform | null {
  if (isIdentity(o)) return null;
  return {rotation: rotationMatrix(o!), origin, translation: o!.translation};
}

/**
 * Fit the dominant near-horizontal plane in a preview sample with RANSAC and
 * return roll/pitch that make it level plus the height shift that puts it at
 * z = 0. Yaw is kept. Returns null when no plane dominates.
 */
export function levelGround(data: Float32Array, current: Orientation): Orientation | null {
  const n = data.length / 6;
  if (n < 100) return null;
  // Work on points already rotated by the current orientation so a second
  // press refines rather than restarts.
  const q = quaternion(current);
  const step = Math.max(1, Math.floor(n / 60000));
  const pts: THREE.Vector3[] = [];
  const v = new THREE.Vector3();
  for (let i = 0; i < n; i += step) pts.push(v.set(data[i * 6], data[i * 6 + 1], data[i * 6 + 2]).applyQuaternion(q).clone());
  let seed = 12345;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
  const pick = () => pts[Math.floor(rnd() * pts.length)];
  let best: {normal: THREE.Vector3; d: number; inliers: number} | null = null;
  const tolerance = 0.06;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), normal = new THREE.Vector3();
  for (let iter = 0; iter < 300; iter++) {
    const p0 = pick(), p1 = pick(), p2 = pick();
    a.subVectors(p1, p0); b.subVectors(p2, p0); normal.crossVectors(a, b);
    if (normal.lengthSq() < 1e-8) continue;
    normal.normalize();
    if (normal.z < 0) normal.negate();
    if (normal.z < 0.7) continue; // walls and hedges are not the ground
    const d = -normal.dot(p0);
    let inliers = 0;
    for (const p of pts) if (Math.abs(normal.dot(p) + d) < tolerance) inliers++;
    if (!best || inliers > best.inliers) best = {normal: normal.clone(), d, inliers};
  }
  if (!best || best.inliers < pts.length * 0.08) return null;
  // Refine with the inlier centroid and covariance for a cleaner normal.
  const inl = pts.filter(p => Math.abs(best!.normal.dot(p) + best!.d) < tolerance);
  const c = new THREE.Vector3();
  for (const p of inl) c.add(p);
  c.divideScalar(inl.length);
  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (const p of inl) { const dx = p.x - c.x, dy = p.y - c.y, dz = p.z - c.z; xx += dx * dx; xy += dx * dy; xz += dx * dz; yy += dy * dy; yz += dy * dz; zz += dz * dz; }
  // Normal of a near-horizontal plane from the least-squares fit z = ax + by + c.
  const det = xx * yy - xy * xy;
  if (Math.abs(det) > 1e-9) {
    const ga = (xz * yy - yz * xy) / det, gb = (yz * xx - xz * xy) / det;
    best.normal.set(-ga, -gb, 1).normalize();
  }
  void zz;
  // Rotation taking the fitted normal to +Z, composed onto the current one.
  const fix = new THREE.Quaternion().setFromUnitVectors(best.normal, new THREE.Vector3(0, 0, 1));
  const total = fix.multiply(q);
  const e = new THREE.Euler().setFromQuaternion(total, 'ZYX');
  const rotation: [number, number, number] = [e.x / deg, e.y / deg, current.rotation[2]];
  // Keep the user's yaw: rebuild with the new roll/pitch and current yaw, then measure ground height.
  const q2 = quaternion({rotation, translation: [0, 0, 0]});
  const ground = c.clone().applyQuaternion(q.clone().invert()).applyQuaternion(q2);
  return {rotation: rotation.map(x => Math.round(x * 100) / 100) as [number, number, number], translation: [current.translation[0], current.translation[1], Math.round(-ground.z * 1000) / 1000]};
}
