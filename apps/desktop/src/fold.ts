// Steps that finish in a second or two are not worth a row of their own.
// They fold into the next real step: its row spins while they run, with a
// note saying which small step is in progress, and the bar starts when the
// real step begins. The pipeline still runs and records them as separate
// stages; only the display changes.
import type {StageState, StageStatus} from './types';

/** hidden stage id -> the visible stage that absorbs it */
export const FOLDED_INTO: Record<string, string> = {
  cameras: 'candidates',
  global: 'export',
  local: 'export',
  blend: 'export',
};
const NOTE: Record<string, string> = {
  cameras: 'calibrating cameras',
  global: 'balancing exposure between photos',
  local: 'balancing exposure within photos',
  blend: 'blending colors',
};

export type Row = StageState & {
  /** real stage ids this row stands for, host last */
  members: string[];
  /** small step in progress before the host starts */
  note?: string;
  /** which member's log to show */
  logStage: string;
};

const FAILED: StageStatus[] = ['failed', 'cancelled', 'incomplete'];

/** Collapse an ordered stage list into display rows. */
export function foldStages(stages: StageState[]): Row[] {
  const ids = new Set(stages.map(s => s.id));
  const rows: Row[] = [];
  const pendingHidden: StageState[] = [];
  for (const s of stages) {
    const host = FOLDED_INTO[s.id];
    if (host && ids.has(host)) { pendingHidden.push(s); continue; }
    const members = [...pendingHidden, s];
    pendingHidden.length = 0;
    rows.push(fold(members));
  }
  // Hidden steps whose host is absent from this run (for example geometry-only) stay visible.
  for (const s of pendingHidden) rows.push(fold([s]));
  return rows;
}

function fold(members: StageState[]): Row {
  const host = members[members.length - 1];
  const hidden = members.slice(0, -1);
  const sumWall = members.reduce((sum, m) => sum + (m.wall_s ?? 0), 0);
  const startedAt = members.map(m => m.startedAt).filter((t): t is number => t != null).sort()[0];
  const base = {members: members.map(m => m.id), logStage: host.id, startedAt};
  const failed = members.find(m => FAILED.includes(m.status));
  if (failed) return {...host, ...base, status: failed.status, wall_s: null, logStage: failed.id};
  if (host.status === 'running') return {...host, ...base};
  const active = hidden.find(m => m.status === 'running');
  if (active) return {...host, ...base, status: 'running', done: undefined, total: undefined, note: NOTE[active.id] ?? active.id, logStage: active.id};
  if (host.status === 'complete' || host.status === 'cached') return {...host, ...base, wall_s: host.status === 'complete' ? sumWall : host.wall_s};
  const waiting = members.find(m => m.status === 'waiting');
  if (waiting) return {...host, ...base, status: 'waiting', waitingFor: waiting.waitingFor};
  // Some small step finished but the host has not started: the row is in between.
  if (hidden.some(m => m.status === 'complete' || m.status === 'cached')) return {...host, ...base, status: 'running', note: 'preparing'};
  return {...host, ...base};
}
