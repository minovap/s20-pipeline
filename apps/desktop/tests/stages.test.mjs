import assert from 'node:assert/strict';
import {test} from 'node:test';
import {stagesFor} from '../src/types.ts';
import {duration, roughRange, runFolderName} from '../src/format.ts';
import {effectiveBox, intersect, nextSliceName, normalize, unionMask} from '../src/viewer/slices.ts';

test('geometry-only job excludes every photo stage', () => {
  const ids = stagesFor({color: false, mask: 'person', exposure: 'local', pose_refinement: true}).map(s => s.id);
  assert.deepEqual(ids, ['decode', 'pack', 'tracking', 'pose_refinement', 'registered', 'geometry']);
});
test('optional stages follow effective processing options', () => {
  const ids = stagesFor({color: true, mask: 'off', exposure: 'off', pose_refinement: false}).map(s => s.id);
  assert(!ids.includes('local')); assert(!ids.includes('global')); assert(!ids.includes('masks')); assert(!ids.includes('pose_refinement'));
  assert(ids.includes('blend')); assert(ids.includes('export'));
});
test('durations never show fractions of a second', () => {
  assert.equal(duration(null), '');
  assert.equal(duration(0.4), '0s');
  assert.equal(duration(38.1), '38s');
  assert.equal(duration(252), '4m 12s');
  assert.equal(duration(3725), '1h 02m 05s');
  assert.equal(roughRange([200, 700]), '3 min to 12 min');
  assert.match(runFolderName(new Date(2026, 8, 11, 18, 4, 33)), /^2026-09-11 18-04-33$/);
});
test('slice boxes intersect along the parent chain', () => {
  const a = [[0, 0, 0], [10, 10, 10]], b = [[5, -5, 2], [15, 5, 20]];
  assert.deepEqual(intersect(a, b), [[5, 0, 2], [10, 5, 10]]);
  assert.equal(intersect(a, [[11, 0, 0], [12, 1, 1]]), null);
  assert.deepEqual(normalize([[3, 2, 1], [0, 0, 0]]), [[0, 0, 0], [3, 2, 1]]);
  const slices = [
    {id: 'p', name: 'Slice 1', source: 's', parent: null, box: a, created: 0},
    {id: 'c', name: 'Slice 1 1', source: 's', parent: 'p', box: b, created: 0},
  ];
  assert.deepEqual(effectiveBox(slices[1], slices), [[5, 0, 2], [10, 5, 10]]);
  assert.equal(nextSliceName(slices), 'Slice 2');
  assert.equal(nextSliceName(slices, slices[0]), 'Slice 1 2');
});
test('union mask shows a point once even when boxes overlap', () => {
  // three points at x = 1, 6, 12 (local), origin 100 on x
  const data = new Float32Array([1, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0]);
  const mask = unionMask(data, [100, 0, 0], [[[100, -1, -1], [107, 1, 1]], [[105, -1, -1], [110, 1, 1]]]);
  assert.deepEqual([...mask], [1, 1, 0]);
});

test('level ground finds a tilted plane and puts it at height 0', async () => {
  const {levelGround, IDENTITY, rotationMatrix} = await import('../src/viewer/orient.ts');
  const n = 4000, data = new Float32Array(n * 6);
  // Plane tilted 5 degrees about x, offset 2 m up, plus some scattered noise points above it.
  const tilt = 5 * Math.PI / 180;
  for (let i = 0; i < n; i++) {
    const x = (i % 63) - 31, y = Math.floor(i / 63) - 31;
    const onPlane = i % 10 !== 0;
    const z = onPlane ? 2 + Math.tan(tilt) * y : 2 + Math.tan(tilt) * y + 1 + (i % 7);
    data.set([x, y, z, 0.5, 0.5, 0.5], i * 6);
  }
  const o = levelGround(data, IDENTITY);
  assert(o, 'plane found');
  assert(Math.abs(Math.abs(o.rotation[0]) - 5) < 0.3, `roll ${o.rotation[0]}`);
  assert(Math.abs(o.rotation[1]) < 0.3, `pitch ${o.rotation[1]}`);
  assert(Math.abs(o.translation[2] + 2) < 0.1, `height ${o.translation[2]}`);
  const r = rotationMatrix({rotation: [0, 0, 90], translation: [0, 0, 0]});
  // yaw 90: x axis maps to y
  assert(Math.abs(r[3] - 1) < 1e-9 && Math.abs(r[1] + 1) < 1e-9);
});
