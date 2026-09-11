import assert from 'node:assert/strict';
import {test} from 'node:test';
import {stagesFor,seconds} from '../src/types.ts';
test('geometry-only job excludes every photo stage',()=>{const stages=stagesFor({color:false,mask:'person',exposure:'local',pose_refinement:true});assert.deepEqual(stages.map(s=>s[0]),['decode','pack','tracking','pose_refinement','registered','geometry']);});
test('optional stages follow effective processing options',()=>{const ids=stagesFor({color:true,mask:'off',exposure:'off',pose_refinement:false}).map(s=>s[0]);assert(!ids.includes('local'));assert(!ids.includes('global'));assert(!ids.includes('masks'));assert(!ids.includes('pose_refinement'));assert(ids.includes('blend'));assert(ids.includes('export'));});
test('unavailable timings never appear as zero',()=>{assert.equal(seconds(null),'—');assert.equal(seconds(undefined),'—');assert.equal(seconds(0),'0.0 s');});
