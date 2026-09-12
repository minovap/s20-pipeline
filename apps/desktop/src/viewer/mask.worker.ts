// Off-thread point masks. The worker keeps a copy of each cloud's positions
// and answers with a visibility mask, the visible bounds and a count, so the
// interface never blocks on a 12 M point polygon test.
import {unionMask, countMask} from './slices.ts';
import type {Region} from './slices.ts';
import type {Box} from '../types';
import type {Transform} from './orient.ts';

type Request =
  | {type: 'load'; source: string; positions: Float32Array}
  | {type: 'unload'; source: string}
  | {type: 'mask'; id: number; source: string; origin: number[]; regions: (Box | Region)[]; transform: Transform | null}
  | {type: 'count'; id: number; source: string; origin: number[]; region: Box | Region; transform: Transform | null};

const clouds = new Map<string, Float32Array>();

function bounds(positions: Float32Array, mask: Float32Array): [number, number, number, number, number, number] | null {
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity, any = false;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    any = true;
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; if (z < z0) z0 = z; if (z > z1) z1 = z;
  }
  return any ? [x0, y0, z0, x1, y1, z1] : null;
}

self.onmessage = (event: MessageEvent<Request>) => {
  const m = event.data;
  if (m.type === 'load') { clouds.set(m.source, m.positions); return; }
  if (m.type === 'unload') { clouds.delete(m.source); return; }
  const positions = clouds.get(m.source);
  if (!positions) { (self as unknown as Worker).postMessage({type: 'missing', id: m.id}); return; }
  if (m.type === 'mask') {
    const mask = unionMask(positions, m.origin, m.regions, m.transform);
    (self as unknown as Worker).postMessage({type: 'mask', id: m.id, source: m.source, mask, count: countMask(mask), bounds: bounds(positions, mask)}, [mask.buffer]);
  } else if (m.type === 'count') {
    const mask = unionMask(positions, m.origin, [m.region], m.transform);
    (self as unknown as Worker).postMessage({type: 'count', id: m.id, count: countMask(mask)});
  }
};
