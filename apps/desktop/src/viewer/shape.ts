// Outline slices: a polygon of fixed size that can be placed and turned over
// the cloud. Vertices are metres relative to the shape's centre; the slice
// stores where that centre sits in the world and how the shape is turned.
// A perimeter band is the region within `expand` metres outside the polygon.
import type {Box, Shape, Slice} from '../types';

export type XY = [number, number];

/** Lyckan 8 from the certified cadastral extract (Danderyd, 1:400, SWEREF 99 18 00).
 *  Five corners, x east and y north in metres from the north-west corner; 2352 m² against the registered 2358 m². */
export const LYCKAN_8: XY[] = [[0, 0], [39.53, -0.47], [43.83, -62.26], [7.54, -58.74], [4.42, -55.49]];
/** Building footprints from the same extract, in the same frame: the house, the annex and a shed. Drawn as placement guides only. */
export const LYCKAN_8_BUILDINGS: XY[][] = [
  [[22.5, -38.2], [22.27, -38.54], [20.46, -38.36], [20.54, -37.56], [17.86, -37.27], [17.66, -39.32], [14.34, -38.98], [13.31, -49.26], [14.53, -49.4], [15.34, -50.45], [17.39, -50.65], [17.22, -52.09], [21.83, -52.55], [21.98, -51.12], [24.05, -51.33], [25.03, -50.53], [26.25, -50.67], [27.45, -38.7]],
  [[14.17, -40.64], [13.48, -47.53], [7.5, -46.94], [8.21, -40.03]],
  [[27.42, -38.88], [31.75, -39.34], [31.34, -42.45], [27.11, -42.03]],
];

export function area(vertices: XY[]): number {
  let sum = 0;
  for (let i = 0; i < vertices.length; i++) {
    const [x0, y0] = vertices[i], [x1, y1] = vertices[(i + 1) % vertices.length];
    sum += x0 * y1 - x1 * y0;
  }
  return Math.abs(sum) / 2;
}

export function centroid(vertices: XY[]): XY {
  let sx = 0, sy = 0;
  for (const [x, y] of vertices) { sx += x; sy += y; }
  return [sx / vertices.length, sy / vertices.length];
}

/** Move the vertices so their centroid is the origin. */
export function centred(vertices: XY[]): XY[] {
  const [cx, cy] = centroid(vertices);
  return vertices.map(([x, y]) => [x - cx, y - cy]);
}
/** Shift extra polylines by the same amount `centred` applied to the outline they belong to. */
export function centredWith(outline: XY[], others: XY[][]): XY[][] {
  const [cx, cy] = centroid(outline);
  return others.map(poly => poly.map(([x, y]) => [x - cx, y - cy] as XY));
}

export function rectangle(areaM2: number, aspect = 1.5): XY[] {
  const h = Math.sqrt(areaM2 / aspect), w = h * aspect;
  return [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]];
}

/** "x y" per line, metres; blank lines and a leading "#" comment are ignored. */
export function parseVertices(text: string): XY[] | null {
  const out: XY[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const parts = line.split(/[\s,;]+/).map(Number);
    if (parts.length < 2 || parts.slice(0, 2).some(v => !isFinite(v))) return null;
    out.push([parts[0], parts[1]]);
  }
  return out.length >= 3 ? out : null;
}
export const formatVertices = (vertices: XY[]) => vertices.map(([x, y]) => `${x.toFixed(2)} ${y.toFixed(2)}`).join('\n');

/** Polygon in world XY after rotation (degrees, counterclockwise) and placement. */
export function worldPolygon(shape: Shape): XY[] {
  return worldPoints(shape, shape.vertices);
}
export function worldPoints(shape: Pick<Shape, 'position' | 'rotation'>, points: XY[]): XY[] {
  const a = (shape.rotation * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return points.map(([x, y]) => [shape.position[0] + x * c - y * s, shape.position[1] + x * s + y * c]);
}

/** Approximate outward offset for display: each vertex moves along its angle bisector. */
export function offsetPolygon(vertices: XY[], distance: number): XY[] {
  const n = vertices.length;
  if (distance === 0) return vertices.map(v => [...v] as XY);
  const sign = area(vertices) > 0 && signedArea(vertices) > 0 ? 1 : -1;
  return vertices.map((p, i) => {
    const prev = vertices[(i + n - 1) % n], next = vertices[(i + 1) % n];
    const e0 = normal(prev, p, sign), e1 = normal(p, next, sign);
    const bx = e0[0] + e1[0], by = e0[1] + e1[1];
    const len = Math.hypot(bx, by) || 1;
    const cosHalf = Math.max(0.3, (bx * e0[0] + by * e0[1]) / len);
    return [p[0] + (bx / len) * (distance / cosHalf), p[1] + (by / len) * (distance / cosHalf)];
  });
}
function signedArea(v: XY[]): number {
  let sum = 0;
  for (let i = 0; i < v.length; i++) { const a = v[i], b = v[(i + 1) % v.length]; sum += a[0] * b[1] - b[0] * a[1]; }
  return sum / 2;
}
/** Outward unit normal of edge a→b for a polygon whose signed area has `sign`. */
function normal(a: XY, b: XY, sign: number): XY {
  const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
  return sign > 0 ? [dy / len, -dx / len] : [-dy / len, dx / len];
}

export function insidePolygon(x: number, y: number, v: XY[]): boolean {
  let inside = false;
  for (let i = 0, j = v.length - 1; i < v.length; j = i++) {
    const [xi, yi] = v[i], [xj, yj] = v[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
export function distanceToEdges(x: number, y: number, v: XY[]): number {
  let best = Infinity;
  for (let i = 0, j = v.length - 1; i < v.length; j = i++) {
    const [ax, ay] = v[j], [bx, by] = v[i];
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
    const t = l2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
    const ex = ax + t * dx - x, ey = ay + t * dy - y;
    best = Math.min(best, ex * ex + ey * ey);
  }
  return Math.sqrt(best);
}

/** One polygon condition of a slice region. */
export type PolygonTest = {vertices: XY[]; expand: number; mode: 'inside' | 'ring'};
export function passes(x: number, y: number, test: PolygonTest): boolean {
  const inside = insidePolygon(x, y, test.vertices);
  if (test.mode === 'inside') return inside;
  return !inside && distanceToEdges(x, y, test.vertices) <= test.expand;
}

/** World-space bounding box of a shape slice over a z range, used for the box chain and export clipping. */
export function shapeBox(shape: Shape, mode: 'inside' | 'ring', expand: number, z: [number, number]): Box {
  const poly = mode === 'ring' ? offsetPolygon(worldPolygon(shape), expand * 1.5) : worldPolygon(shape);
  const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
  return [[Math.min(...xs), Math.min(...ys), z[0]], [Math.max(...xs), Math.max(...ys), z[1]]];
}

/** The shape a slice draws or tests: its own, or its parent's when it is a perimeter band. */
export function shapeOf(slice: Slice, all: Slice[]): Shape | null {
  if (slice.shape) return slice.shape;
  if (slice.ring && slice.parent) return all.find(s => s.id === slice.parent)?.shape ?? null;
  return null;
}
