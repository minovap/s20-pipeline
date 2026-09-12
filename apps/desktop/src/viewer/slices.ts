// Pure slice geometry. Boxes are axis-aligned and stored in the source
// cloud's own coordinates, so the same numbers drive display and export.
import type {Box, Slice} from '../types';
import type {Transform} from './orient';

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

/**
 * Points inside any box are visible. Positions are xyz triples local to
 * `origin`; with a transform, world = R·local + origin + t.
 */
export function unionMask(data: Float32Array, origin: number[], boxes: Box[], transform?: Transform | null, out?: Float32Array): Float32Array {
  const n = data.length / 3;
  const mask = out && out.length === n ? out : new Float32Array(n);
  mask.fill(0);
  const shift = transform ? [origin[0] + transform.translation[0], origin[1] + transform.translation[1], origin[2] + transform.translation[2]] : origin;
  const local = boxes.map(b => [
    b[0][0] - shift[0], b[0][1] - shift[1], b[0][2] - shift[2],
    b[1][0] - shift[0], b[1][1] - shift[1], b[1][2] - shift[2],
  ]);
  const r = transform?.rotation;
  for (let i = 0; i < n; i++) {
    let x = data[i * 3], y = data[i * 3 + 1], z = data[i * 3 + 2];
    if (r) { const px = x, py = y, pz = z; x = r[0] * px + r[1] * py + r[2] * pz; y = r[3] * px + r[4] * py + r[5] * pz; z = r[6] * px + r[7] * py + r[8] * pz; }
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
