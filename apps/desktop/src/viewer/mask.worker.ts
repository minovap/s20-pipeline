// Off-thread point masks. The worker keeps a copy of each cloud's positions
// and answers with a visibility mask, the visible bounds and a count, so the
// interface never blocks on a 12 M point polygon test.
import {unionMask, countMask} from './slices.ts';
import type {Region} from './slices.ts';
import type {Box} from '../types';
import type {Transform} from './orient.ts';

/** A region with a cache key: the same key always means the same region on the same cloud. */
type Keyed = {key: string; region: Box | Region};
type Request =
  | {type: 'load'; source: string; positions: Float32Array}
  | {type: 'unload'; source: string}
  | {type: 'mask'; id: number; source: string; origin: number[]; regions: Keyed[]; transform: Transform | null; transformKey: string}
  | {type: 'count'; id: number; source: string; origin: number[]; region: Keyed; transform: Transform | null; transformKey: string};

const clouds = new Map<string, Float32Array>();
/** Cached per-region masks: source -> cache key -> mask. Cleared with the cloud. */
const cache = new Map<string, Map<string, Float32Array>>();
const CACHE_LIMIT = 24;

function regionMask(source: string, positions: Float32Array, origin: number[], keyed: Keyed, transform: Transform | null, transformKey: string): Float32Array {
  // A page and worker from different hot reloads may disagree on the shape; accept a bare region too.
  if (!keyed || typeof keyed !== 'object' || !('region' in keyed)) keyed = {key: JSON.stringify(keyed), region: keyed as unknown as Box | Region};
  const key = `${transformKey}|${keyed.key}`;
  let perSource = cache.get(source);
  if (!perSource) { perSource = new Map(); cache.set(source, perSource); }
  const hit = perSource.get(key);
  if (hit) return hit;
  const mask = unionMask(positions, origin, [keyed.region], transform);
  if (perSource.size >= CACHE_LIMIT) perSource.delete(perSource.keys().next().value!);
  perSource.set(key, mask);
  return mask;
}

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
  if (m.type === 'load') { clouds.set(m.source, m.positions); cache.delete(m.source); return; }
  if (m.type === 'unload') { clouds.delete(m.source); cache.delete(m.source); return; }
  const positions = clouds.get(m.source);
  if (!positions) { (self as unknown as Worker).postMessage({type: 'missing', id: m.id}); return; }
  if (m.type === 'mask') {
    // Union of cached per-region masks: a toggle after the first computation costs one pass of ORs.
    const parts = m.regions.map(r => regionMask(m.source, positions, m.origin, r, m.transform, m.transformKey));
    const mask = new Float32Array(positions.length / 3);
    for (const part of parts) for (let i = 0; i < mask.length; i++) if (part[i]) mask[i] = 1;
    (self as unknown as Worker).postMessage({type: 'mask', id: m.id, source: m.source, mask, count: countMask(mask), bounds: bounds(positions, mask)}, [mask.buffer]);
  } else if (m.type === 'count') {
    const mask = regionMask(m.source, positions, m.origin, m.region, m.transform, m.transformKey);
    (self as unknown as Worker).postMessage({type: 'count', id: m.id, count: countMask(mask)});
  }
};
