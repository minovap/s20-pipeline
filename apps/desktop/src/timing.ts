// Time-remaining estimates. Running steps with counts use their recent rate;
// everything else uses this Mac's history of the same step on similar scans.
import type {RunSize, StageState} from './types';

// Mirrors GEOMETRY_LANE in s20_pipeline/runner.py; kept local so this module has no runtime imports.
const GEOMETRY_LANE = new Set(['copy', 'decode', 'pack', 'tracking', 'pose_refinement', 'registered', 'geometry']);

type Entry = {metric: number; wall: number};
type History = Record<string, Entry[]>;
const KEY = 'timing:v1';
const KEEP = 12;

/** The size measure a step's duration scales with. */
export function metricFor(stage: string, size: RunSize): number {
  if (stage === 'copy') return 0;
  if (['photos', 'masks', 'cameras'].includes(stage)) return Math.max(1, size.photos);
  if (['candidates', 'global', 'local', 'blend'].includes(stage)) return Math.max(1, size.frames) * Math.max(1, size.photos);
  return Math.max(1, size.frames);
}

export function loadHistory(): History {
  try { return JSON.parse(localStorage.getItem(KEY) ?? '{}'); } catch { return {}; }
}
export function recordTiming(stage: string, size: RunSize, wall: number, history = loadHistory()): History {
  const metric = metricFor(stage, size);
  if (!metric || !isFinite(wall) || wall <= 0) return history;
  const next = {...history, [stage]: [...(history[stage] ?? []), {metric, wall}].slice(-KEEP)};
  try { localStorage.setItem(KEY, JSON.stringify(next)); } catch { /* history is a convenience */ }
  return next;
}

/** Predicted wall seconds for a step on a scan of this size, or null without history. Median of wall/metric ratios. */
export function predict(stage: string, size: RunSize, history: History): number | null {
  const entries = history[stage];
  const metric = metricFor(stage, size);
  if (!entries?.length || !metric) return null;
  const ratios = entries.map(e => e.wall / e.metric).sort((a, b) => a - b);
  const median = ratios[Math.floor(ratios.length / 2)];
  return median * metric;
}

/** Seconds left for a running step from its recent rate, or null before enough progress. */
export function remainingFromRate(stage: StageState, now: number): number | null {
  const samples = stage.samples ?? [];
  if (!stage.total || stage.done == null || samples.length < 2) return null;
  const elapsed = stage.startedAt ? (now - stage.startedAt) / 1000 : 0;
  if (elapsed < 10 || stage.done / stage.total < 0.05) return null;
  const recent = samples.filter(([t]) => now - t <= 30000);
  const window = recent.length >= 2 ? recent : samples.slice(-2);
  const [t0, d0] = window[0], [t1, d1] = window[window.length - 1];
  if (t1 <= t0 || d1 <= d0) return null;
  const rate = (d1 - d0) / ((t1 - t0) / 1000);
  return (stage.total - stage.done) / rate;
}

/** Seconds left for one step: rate while running, history otherwise. */
export function remainingFor(stage: StageState, size: RunSize, history: History, now: number): number | null {
  if (stage.status === 'complete' || stage.status === 'cached' || stage.status === 'skipped') return 0;
  if (stage.status === 'running') {
    const byRate = remainingFromRate(stage, now);
    if (byRate != null) return byRate;
    const predicted = predict(stage.id, size, history);
    if (predicted == null) return null;
    const elapsed = stage.startedAt ? (now - stage.startedAt) / 1000 : 0;
    return Math.max(predicted - elapsed, predicted * 0.1);
  }
  return predict(stage.id, size, history);
}

/** Time left for the whole run: the longer of the two parallel lanes plus the serial tail. Null when any pending step has no prediction. */
export function remainingTotal(stages: StageState[], size: RunSize, history: History, now: number): number | null {
  const prep = new Set(['photos', 'masks', 'cameras']);
  let geometry = 0, color = 0, tail = 0;
  for (const s of stages) {
    const left = remainingFor(s, size, history, now);
    if (left == null) return null;
    if (GEOMETRY_LANE.has(s.id)) geometry += left;
    else if (prep.has(s.id)) color += left;
    else tail += left;
  }
  return Math.max(geometry, color) + tail;
}

/** "about 12 min left", "under a minute left", "about 1 h 20 min left". */
export function leftText(seconds: number | null): string {
  if (seconds == null) return '';
  if (seconds < 60) return 'under a minute left';
  const m = Math.round(seconds / 60);
  if (m < 60) return `about ${m} min left`;
  return `about ${Math.floor(m / 60)} h ${m % 60 ? `${m % 60} min ` : ''}left`;
}
