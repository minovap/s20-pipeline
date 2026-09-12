// Pure slice geometry. Boxes are axis-aligned and stored in the source
// cloud's own coordinates, so the same numbers drive display and export.
import type {Box, Slice} from '../types';
import type {Transform} from './orient';
import {passes, shapeOf, worldPolygon} from './shape.ts';
import type {PolygonTest} from './shape.ts';

export function intersect(a: Box, b: Box): Box | null {
  const lo: [number, number, number] = [Math.max(a[0][0], b[0][0]), Math.max(a[0][1], b[0][1]), Math.max(a[0][2], b[0][2])];
  const hi: [number, number, number] = [Math.min(a[1][0], b[1][0]), Math.min(a[1][1], b[1][1]), Math.min(a[1][2], b[1][2])];
  for (let i = 0; i < 3; i++) if (lo[i] > hi[i]) return null;
  return [lo, hi];
}

export function normalize(box: Box): Box {
  return [
    [Math.min(box[0][0], box[1][0]), Math.min(box[0][1], box[1][1]), Math.min(box[0][2], box[1][2])],
    [Math.max(box[0][0], box[1][0]), Math.max(box[0][1], box[1][1]), Math.max(box[0][2], box[1][2])],
  ];
}

/** Everything a slice's points must satisfy: the clipped box plus polygon tests up the chain. */
export type Region = {box: Box; tests: PolygonTest[]};
export function effectiveRegion(slice: Slice, all: Slice[]): Region | null {
  const box = effectiveBox(slice, all);
  if (!box) return null;
  const tests: PolygonTest[] = [];
  let current: Slice | undefined = slice;
  let skipInside: string | null = null; // a ring replaces its parent's inside test
  let guard = 0;
  while (current && guard++ < 64) {
    if (current.ring && current.parent) {
      const shape = shapeOf(current, all);
      if (shape) tests.push({vertices: worldPolygon(shape), expand: current.ring.expand, mode: 'ring'});
      skipInside = current.parent;
    } else if (current.shape && current.id !== skipInside) {
      tests.push({vertices: worldPolygon(current.shape), expand: 0, mode: 'inside'});
    }
    current = current.parent ? all.find(s => s.id === current!.parent) : undefined;
  }
  return {box, tests};
}

/** The region a slice really covers: its own box clipped by every ancestor. A perimeter band lies outside its parent outline, so that parent's box does not clip it. */
export function effectiveBox(slice: Slice, all: Slice[]): Box | null {
  let box: Box | null = slice.box;
  let parent = slice.parent;
  let skip = slice.ring ? slice.parent : null;
  let guard = 0;
  while (box && parent && guard++ < 64) {
    const p = all.find(s => s.id === parent);
    if (!p) break;
    if (p.id !== skip) box = intersect(box, p.box);
    parent = p.parent;
  }
  return box;
}

export function children(id: string, all: Slice[]): Slice[] {
  return all.filter(s => s.parent === id);
}
export function descendants(id: string, all: Slice[]): Slice[] {
  const out: Slice[] = [];
  const walk = (parent: string) => { for (const c of children(parent, all)) { out.push(c); walk(c.id); } };
  walk(id);
  return out;
}

/**
 * Points inside any box are visible. Positions are xyz triples local to
 * `origin`; with a transform, world = R·local + origin + t.
 */
export function unionMask(data: Float32Array, origin: number[], regions: (Box | Region)[], transform?: Transform | null, out?: Float32Array): Float32Array {
  const n = data.length / 3;
  const mask = out && out.length === n ? out : new Float32Array(n);
  mask.fill(0);
  const shift = transform ? [origin[0] + transform.translation[0], origin[1] + transform.translation[1], origin[2] + transform.translation[2]] : origin;
  const normalized = regions.map(r => (Array.isArray(r) ? {box: r as Box, tests: [] as PolygonTest[]} : r));
  const local = normalized.map(({box: b, tests}) => ({
    lo: [b[0][0] - shift[0], b[0][1] - shift[1], b[0][2] - shift[2]],
    hi: [b[1][0] - shift[0], b[1][1] - shift[1], b[1][2] - shift[2]],
    tests,
  }));
  const r = transform?.rotation;
  for (let i = 0; i < n; i++) {
    let x = data[i * 3], y = data[i * 3 + 1], z = data[i * 3 + 2];
    if (r) { const px = x, py = y, pz = z; x = r[0] * px + r[1] * py + r[2] * pz; y = r[3] * px + r[4] * py + r[5] * pz; z = r[6] * px + r[7] * py + r[8] * pz; }
    for (const b of local) {
      if (x < b.lo[0] || x > b.hi[0] || y < b.lo[1] || y > b.hi[1] || z < b.lo[2] || z > b.hi[2]) continue;
      let ok = true;
      for (const test of b.tests) if (!passes(x + shift[0], y + shift[1], test)) { ok = false; break; }
      if (ok) { mask[i] = 1; break; }
    }
  }
  return mask;
}

export function countMask(mask: Float32Array): number {
  let c = 0;
  for (let i = 0; i < mask.length; i++) c += mask[i];
  return c;
}

/** Default names count upward per source: "Slice 1", "Slice 2". */
export function nextSliceName(all: Slice[], parent?: Slice | null): string {
  if (parent) {
    const n = children(parent.id, all).length + 1;
    return `${parent.name} ${n}`;
  }
  let n = all.filter(s => !s.parent).length + 1;
  const names = new Set(all.map(s => s.name));
  while (names.has(`Slice ${n}`)) n++;
  return `Slice ${n}`;
}

export const newId = () => (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2) + Date.now().toString(36));
