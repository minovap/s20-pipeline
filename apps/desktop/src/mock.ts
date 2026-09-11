// Browser-only stand-in for the Rust shell, used to review the UI without
// Tauri: open the Vite dev server with ?mock. Nothing here ships in the app
// path when running inside Tauri.
import type {Box, Job, Preview, Project, ProjectSummary, Run, Settings} from './types';

type Listener = (payload: unknown) => void;
const listeners = new Map<string, Set<Listener>>();
export function emit(name: string, payload: unknown) { listeners.get(name)?.forEach(fn => fn(payload)); }
export function mockListen(name: string, fn: Listener) {
  if (!listeners.has(name)) listeners.set(name, new Set());
  listeners.get(name)!.add(fn);
  return () => { listeners.get(name)?.delete(fn); };
}

const now = Date.now() / 1000;
const root = '/Users/demo/Documents/S20 Projects';
const completedRun: Run = {
  path: `${root}/Garden/runs/2026-09-10 14-02-11`, name: '2026-09-10 14-02-11', status: 'completed', stage: null, error: null,
  started: now - 86400 - 2000, finished: now - 86400 - 2000 + 1493, capture: '/Volumes/SD_CARD/Garden-2026-09-10',
  options: {color: true, mask: 'person', exposure: 'local', pose_refinement: true, resources: 'throughput', memory_gb: 16},
  stages: [['decode', 41], ['pack', 12], ['tracking', 318], ['pose_refinement', 96], ['registered', 8], ['geometry', 64], ['photos', 22], ['cameras', 5], ['masks', 71], ['candidates', 402], ['global', 39], ['local', 188], ['blend', 201], ['export', 26]].map(([id, w]) => ({id: id as string, status: 'complete' as const, wall_s: w as number})),
  result: `${root}/Garden/runs/2026-09-10 14-02-11/export/colorized.las`, result_points: 6021344,
};
const cancelledRun: Run = {
  ...completedRun, path: `${root}/Garden/runs/2026-09-11 09-30-00`, name: '2026-09-11 09-30-00', status: 'cancelled', stage: 'tracking', error: 'Cancelled by user',
  started: now - 7200, finished: now - 7200 + 140, stages: [{id: 'decode', status: 'complete', wall_s: 40}, {id: 'pack', status: 'complete', wall_s: 11}, {id: 'tracking', status: 'incomplete', wall_s: null}], result: null, result_points: null,
};
let project: Project = {
  path: `${root}/Garden`, name: 'Garden', created: now - 200000,
  inputs: [{path: '/Volumes/SD_CARD/Garden-2026-09-10', name: 'Garden-2026-09-10', added: now - 190000, estimate: {estimated_seconds: 900, range_seconds: [540, 2250], confidence: 'low'},
    capture: {capture: '/Volumes/SD_CARD/Garden-2026-09-10', bag_bytes: 583e6, bag_duration_s: 69.4, photos: 64, lidar_frames: 671, imu_samples: 13414, device: {device_model: 'S20', lidar_model: 'MID-360', work_duration: 60}, cameras: {left: {width: 3504, height: 4672}}}}],
  clouds: [], exports: [{path: `${root}/Garden/exports/Hedge west.las`, name: 'Hedge west', bytes: 41e6, modified: now - 3000}],
  slices: [
    {id: 'a', name: 'Slice 1', source: completedRun.result!, parent: null, box: [[-12, -8, -1], [4, 6, 3]], created: now - 5000},
    {id: 'b', name: 'Slice 1 1', source: completedRun.result!, parent: 'a', box: [[-12, -8, 0.8], [4, 6, 1.6]], created: now - 4000},
  ],
  runs: [cancelledRun, completedRun],
  orientations: {},
};
const other: ProjectSummary = {path: `${root}/Driveway`, name: 'Driveway', created: now - 900000, input_count: 1, run_count: 3, last_run: {...completedRun, status: 'failed', started: now - 400000}};

function synthCloud(seed: number, n: number): Float32Array {
  const data = new Float32Array(n * 6);
  let s = seed;
  const rnd = () => { s = (s * 1664525 + 1013904223) % 4294967296; return s / 4294967296; };
  for (let i = 0; i < n; i++) {
    const kind = rnd();
    let x, y, z, r, g, b;
    if (kind < 0.62) { x = (rnd() - 0.5) * 34; y = (rnd() - 0.5) * 26; z = 0.02 * Math.sin(x) + (rnd() - 0.5) * 0.05; const v = 0.35 + rnd() * 0.25; r = v * 0.7; g = v; b = v * 0.45; }
    else if (kind < 0.8) { const t = Math.floor(rnd() * 5); const cx = [-10, -4, 3, 9, 12][t], cy = [-6, 5, -3, 6, -8][t]; const h = rnd() * 4 + 1.5; const rad = (1 - Math.abs(h - 3.5) / 3.5) * 1.8 + 0.3; const a = rnd() * Math.PI * 2; x = cx + Math.cos(a) * rad * Math.sqrt(rnd()); y = cy + Math.sin(a) * rad * Math.sqrt(rnd()); z = h; r = 0.15 + rnd() * 0.1; g = 0.35 + rnd() * 0.25; b = 0.12; }
    else if (kind < 0.93) { const along = rnd(); x = -17 + along * 34; y = 13 + (rnd() - 0.5) * 0.3; z = rnd() * 1.6; r = 0.2; g = 0.32 + rnd() * 0.15; b = 0.15; }
    else { x = 12 + rnd() * 6; y = -13 + rnd() * 9; z = rnd() * 3; const v = 0.6 + rnd() * 0.3; r = v; g = v * 0.9; b = v * 0.8; }
    data.set([x, y, z, r, g, b], i * 6);
  }
  return data;
}
const previews = new Map<string, Float32Array>();

let timers: ReturnType<typeof setTimeout>[] = [];
function simulateJob(job: Job) {
  const stages = ['decode', 'pack', 'tracking', 'pose_refinement', 'registered', 'geometry', 'photos', 'cameras', 'masks', 'candidates', 'global', 'local', 'blend', 'export']
    .filter(s => job.color || ['decode', 'pack', 'tracking', 'pose_refinement', 'registered', 'geometry'].includes(s))
    .filter(s => job.pose_refinement || s !== 'pose_refinement').filter(s => job.mask === 'person' || s !== 'masks')
    .filter(s => job.exposure !== 'off' || (s !== 'global' && s !== 'local')).filter(s => job.exposure === 'local' || s !== 'local');
  let t = 300;
  const send = (delay: number, payload: Record<string, unknown>) => timers.push(setTimeout(() => emit('pipeline-event', {run_id: job.output, time_unix: Date.now() / 1000, ...payload}), delay));
  // Color preparation runs beside the geometry lane in the real runner.
  const side = ['photos', 'masks'].filter(s => stages.includes(s));
  let ts = 300;
  for (const stage of side) { send(ts, {event: 'stage_started', stage}); ts += 2200; send(ts, {event: 'stage_completed', stage, wall_s: 2.2}); ts += 50; }
  if (side.length) send(ts, {event: 'stage_waiting', stage: 'cameras', waiting_for: ['registered']});
  stages.forEach((stage, i) => {
    if (side.includes(stage)) return;
    const length = 1800 + (i * 977) % 2600;
    if (job.resume && i < 2) { send(t, {event: 'stage_cached', stage}); t += 200; return; }
    send(t, {event: 'stage_started', stage});
    if (stage === 'masks' || stage === 'candidates' || stage === 'photos') for (let k = 1; k <= 8; k++) send(t + (length * k) / 9, {event: 'progress', stage, done: k * 8, total: 64});
    for (let k = 1; k <= 3; k++) send(t + (length * k) / 4, {event: 'resources', stage, cpu_core_equivalents: 2 + (k * 1.7) % 6, rss_bytes: 1.2e9 + k * 4e8});
    t += length;
    send(t, {event: 'stage_completed', stage, wall_s: length / 1000});
    t += 50;
  });
  send(t, {event: 'completed'});
  timers.push(setTimeout(() => {
    const run: Run = {...completedRun, path: job.output, name: job.output.split('/').pop()!, started: Date.now() / 1000 - t / 1000, finished: Date.now() / 1000, capture: job.capture, options: job, result: `${job.output}/export/colorized.las`, result_points: 6021344};
    project = {...project, runs: [run, ...project.runs.filter(r => r.path !== job.output)]};
    emit('job-exit', {run_id: job.output, code: 0});
  }, t + 100));
}

export const mockApi = {
  settings: async (): Promise<Settings> => ({engine_root: '/Users/demo/s20-pipeline', engine_ready: true, projects_root: root, running: false}),
  configure: async () => {}, setProjectsRoot: async () => {}, hardware: async () => ({cpu_model: 'Apple M4 Max', logical_cpu_cores: 16, memory_bytes: 64e9, available_memory_bytes: 30e9, machine: 'arm64'}),
  reveal: async () => {},
  listProjects: async (): Promise<ProjectSummary[]> => [{path: project.path, name: project.name, created: project.created, input_count: project.inputs.length, run_count: project.runs.length, last_run: project.runs[0] ?? null}, other],
  createProject: async (name: string) => ({...project, path: `${root}/${name}`, name, inputs: [], runs: [], exports: [], slices: [], clouds: []}),
  openProject: async () => project,
  writeProject: async (_: string, patch: Partial<Project>) => { project = {...project, ...patch}; return project; },
  addInput: async () => project, inputAvailable: async () => true,
  readStageLog: async (_: string, stage: string) => `[${stage}] worker started\n[${stage}] 671 frames\n[${stage}] done`,
  deleteRun: async (run: string) => { project = {...project, runs: project.runs.filter(r => r.path !== run)}; },
  deleteExport: async (path: string) => { project = {...project, exports: project.exports.filter(x => x.path !== path)}; },
  startJob: async (job: Job) => { timers.forEach(clearTimeout); timers = []; simulateJob(job); },
  cancelJob: async () => { timers.forEach(clearTimeout); timers = []; },
  exportSlices: async (spec: {output: string; sources: {path: string; boxes: Box[]}[]}) => {
    const id = 'x' + Date.now();
    const name = spec.output.split('/').pop()!.replace(/\.las$/, '');
    for (let k = 1; k <= 5; k++) setTimeout(() => emit('export-event', {id, name, event: 'progress', done: k, total: 5}), k * 300);
    setTimeout(() => { project = {...project, exports: [{path: spec.output, name, bytes: 12e6, modified: Date.now() / 1000}, ...project.exports]}; emit('export-event', {id, name, event: 'completed', file: spec.output, points: 812331}); }, 1700);
    return id;
  },
  cancelExport: async () => {},
  loadPreview: async (source: string, budget: number): Promise<Preview> => {
    await new Promise(r => setTimeout(r, 300));
    const n = Math.min(budget, 400000);
    const data = synthCloud(source.length, n);
    previews.set(source, data);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], data[i * 6 + k]); hi[k] = Math.max(hi[k], data[i * 6 + k]); }
    return {key: source, name: source.split('/').pop()!, source, source_points: 6021344, display_points: n, origin: [0, 0, 0], bounds: [lo, hi], bytes: n * 24};
  },
  readPreview: async (key: string) => previews.get(key)!.buffer as ArrayBuffer,
};
