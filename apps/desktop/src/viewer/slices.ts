// Pure slice geometry. Boxes are axis-aligned and stored in the source
// cloud's own coordinates, so the same numbers drive display and export.
import type {Box, Slice} from '../types';

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

/** The region a slice really covers: its own box clipped by every ancestor. */
export function effectiveBox(slice: Slice, all: Slice[]): Box | null {
  let box: Box | null = slice.box;
  let parent = slice.parent;
  let guard = 0;
  while (box && parent && guard++ < 64) {
    const p = all.find(s => s.id === parent);
    if (!p) break;
    box = intersect(box, p.box);
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

/** Points inside any box are visible. Positions are interleaved (x,y,z,r,g,b) and local to `origin`. */
export function unionMask(data: Float32Array, origin: number[], boxes: Box[], out?: Float32Array): Float32Array {
  const n = data.length / 6;
  const mask = out && out.length === n ? out : new Float32Array(n);
  mask.fill(0);
  const local = boxes.map(b => [
    b[0][0] - origin[0], b[0][1] - origin[1], b[0][2] - origin[2],
    b[1][0] - origin[0], b[1][1] - origin[1], b[1][2] - origin[2],
  ]);
  for (let i = 0; i < n; i++) {
    const x = data[i * 6], y = data[i * 6 + 1], z = data[i * 6 + 2];
    for (const b of local) {
      if (x >= b[0] && x <= b[3] && y >= b[1] && y <= b[4] && z >= b[2] && z <= b[5]) { mask[i] = 1; break; }
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
