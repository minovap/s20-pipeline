// Build-pipeline style step list. The running step is kept in the vertical
// centre of the scroll area; times never show fractions of a second.
import React, {useEffect, useMemo, useRef, useState} from 'react';
import {Check, ChevronDown, ChevronRight, Minus, X} from 'lucide-react';
import {STAGES, stageLabel} from './types';
import type {RunSize, StageState} from './types';
import {bytes, count, duration} from './format';
import {Spinner} from './ui';
import {leftText, loadHistory, remainingFor, remainingTotal} from './timing';
import {foldStages} from './fold';
import type {Row} from './fold';

export type PipelineProps = {
  stages: StageState[];            // ordered: every step this run will do
  status: string;                  // running | completed | failed | cancelled | interrupted | starting
  startedAt: number | null;        // unix ms
  finishedAt: number | null;       // unix ms
  error?: string | null;
  cpu?: number | null;             // core equivalents while running
  memory?: number | null;          // rss bytes while running
  size?: RunSize;                  // scan size, for time predictions
  onShowLog?: (stage: string) => Promise<string>;
};

const formatProgress = (s: StageState) => {
  if (!s.total) return '';
  const value = s.unit === 'bytes' ? `${bytes(s.done)} of ${bytes(s.total)}` : `${count(s.done)} of ${count(s.total)}${s.unit ? ` ${s.unit}` : ''}`;
  return s.phase ? `${s.phase}, ${value}` : value;
};

export function Pipeline({stages: realStages, status, startedAt, finishedAt, error, cpu, memory, size, onShowLog}: PipelineProps) {
  const [now, setNow] = useState(Date.now());
  const history = useMemo(loadHistory, [status]);
  const live = status === 'running' || status === 'starting';
  // Small steps fold into the next real one for display; timing still uses the real stages.
  const stages: Row[] = useMemo(() => foldStages(realStages), [realStages]);
  const byId = useMemo(() => new Map(realStages.map(s => [s.id, s])), [realStages]);
  const left = (row: Row) => {
    if (!live || !size) return null;
    let sum = 0;
    for (const id of row.members) {
      const member = byId.get(id);
      const value = member ? remainingFor(member, size, history, now) : null;
      if (value == null) return null;
      sum += value;
    }
    return sum;
  };
  const totalLeft = live && size ? remainingTotal(realStages, size, history, now) : null;
  useEffect(() => {
    if (status !== 'running' && status !== 'starting') return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [status]);

  const list = useRef<HTMLDivElement>(null);
  const runningIds = stages.filter(s => s.status === 'running').map(s => s.id);
  const activeId = runningIds[0] ?? null;
  const userScrolled = useRef(false);
  // Follow the active step unless the user has scrolled since it changed.
  useEffect(() => {
    userScrolled.current = false;
    if (!activeId || !list.current) return;
    const row = list.current.querySelector<HTMLElement>(`[data-stage="${activeId}"]`);
    row?.scrollIntoView({block: 'center', behavior: 'smooth'});
  }, [activeId]);
  useEffect(() => {
    if (!activeId || !list.current || userScrolled.current) return;
    const row = list.current.querySelector<HTMLElement>(`[data-stage="${activeId}"]`);
    row?.scrollIntoView({block: 'center', behavior: 'smooth'});
  }, [stages.length]);

  const done = stages.filter(s => s.status === 'complete' || s.status === 'cached').length;
  const total = stages.length;
  const end = finishedAt ?? now;
  const elapsed = startedAt ? (end - startedAt) / 1000 : null;
  const position = activeId ? stages.findIndex(s => s.id === activeId) + 1 : done;
  const positions = runningIds.map(id => stages.findIndex(s => s.id === id) + 1);

  const headline =
    status === 'running' ? (positions.length > 1 ? `Steps ${positions.join(' and ')} of ${total}` : `Step ${position} of ${total}`) :
    status === 'starting' ? 'Checking inputs' :
    status === 'completed' ? `${total} steps completed` :
    status === 'failed' ? `Failed at step ${Math.min(position + 1, total)} of ${total}` :
    status === 'cancelled' ? `Cancelled after ${done} of ${total} steps` :
    status === 'interrupted' ? `Interrupted after ${done} of ${total} steps` : '';

  const [openLog, setOpenLog] = useState<{stage: string; text: string} | null>(null);
  const [openInfo, setOpenInfo] = useState<string | null>(null);
  // The running step's note folds out on its own; a click still toggles any step.
  useEffect(() => { if (activeId) setOpenInfo(activeId); }, [activeId]);
  async function toggleLog(stage: string) {
    if (openLog?.stage === stage) { setOpenLog(null); return; }
    if (!onShowLog) return;
    const source = stages.find(r => r.id === stage)?.logStage ?? stage;
    try { setOpenLog({stage, text: await onShowLog(source)}); } catch (e) { setOpenLog({stage, text: String(e)}); }
  }

  // Group boundaries so the connector line breaks between sections.
  const groupOf = (id: string) => STAGES.find(x => x.id === id)?.group ?? '';
  const groupTime = (group: string) => stages.filter(s => groupOf(s.id) === group).reduce((sum, s) => {
    if (s.status === 'complete') return sum + (s.wall_s ?? 0);
    if (s.status === 'running' && s.startedAt) return sum + (now - s.startedAt) / 1000;
    return sum;
  }, 0);
  const groupDone = (group: string) => stages.filter(s => groupOf(s.id) === group).every(s => s.status === 'complete' || s.status === 'cached' || s.status === 'skipped');

  return (
    <div className={`pipeline ${status}`}>
      <div className="pipeline-head">
        <div>
          <strong>{headline}</strong>
          {(status === 'running') && cpu != null && <span className="live">{cpu.toFixed(1)} cores, {bytes(memory)}</span>}
        </div>
        <div className="totals"><time>{duration(elapsed)}</time>{totalLeft != null && <span className="left">{leftText(totalLeft)}</span>}</div>
      </div>
      {error && !stages.some(x => x.status === 'failed' || x.status === 'cancelled' || x.status === 'incomplete') && <div className="step-error top">{error}</div>}
      <div className="steps" ref={list} onWheel={() => { userScrolled.current = true; }} onTouchMove={() => { userScrolled.current = true; }}>
        {stages.map((s, i) => {
          const group = groupOf(s.id);
          const header = i === 0 || groupOf(stages[i - 1].id) !== group ? group : null;
          const last = i === stages.length - 1 || groupOf(stages[i + 1].id) !== group;
          const live = s.status === 'running' && s.startedAt ? (now - s.startedAt) / 1000 : null;
          const stepLeft = s.status === 'running' || s.status === 'pending' || s.status === 'waiting' ? left(s) : null;
          const time =
            s.status === 'running' ? duration(live) :
            s.status === 'waiting' ? `waiting for ${(s.waitingFor ?? []).map(stageLabel).join(', ').toLowerCase()}` :
            s.status === 'cached' ? 'cached' :
            s.status === 'complete' ? duration(s.wall_s) :
            s.status === 'skipped' ? 'skipped' : '';
          const failed = s.status === 'failed' || s.status === 'cancelled' || s.status === 'incomplete';
          return (
            <React.Fragment key={s.id}>
              {header && (
                <div className={'group' + (groupDone(group) ? ' done' : '')}>
                  <span>{header}</span>
                  <time>{groupTime(group) > 0 ? duration(groupTime(group)) : ''}</time>
                </div>
              )}
              <div className={`step ${s.status}${header ? ' first' : ''}${last ? ' last' : ''}${openInfo === s.id ? ' open' : ''}`} data-stage={s.id}
                role="button" tabIndex={0} aria-expanded={openInfo === s.id} onClick={() => setOpenInfo(o => (o === s.id ? null : s.id))}
                onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpenInfo(o => (o === s.id ? null : s.id)); } }}>
                <span className="mark">
                  {s.status === 'running' ? <Spinner /> :
                   s.status === 'complete' || s.status === 'cached' ? <Check size={13} strokeWidth={2.5} /> :
                   failed ? <X size={13} strokeWidth={2.5} /> :
                   s.status === 'skipped' ? <Minus size={12} /> : null}
                </span>
                <span className="name">{stageLabel(s.id)}
                  {s.status === 'running' && (s.total ? <small>{formatProgress(s)}</small> : s.note ? <small>{s.note}</small> : null)}
                </span>
                <span className="times">
                  <time>{time}</time>
                  {stepLeft != null && s.status !== 'waiting' && <small className="eta">{s.status === 'running' ? `~${duration(Math.round(stepLeft))} left` : `~${duration(Math.round(stepLeft))}`}</small>}
                </span>
                {onShowLog && (s.status !== 'pending' && s.status !== 'skipped') && (
                  <button className="icon log" title="Show log" aria-expanded={openLog?.stage === s.id} onClick={e => { e.stopPropagation(); toggleLog(s.id); }}>
                    {openLog?.stage === s.id ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  </button>
                )}
                {s.status === 'running' && (s.total ? <div className="bar"><i style={{width: `${Math.min(100, (100 * (s.done ?? 0)) / s.total)}%`}} /></div> : <div className="bar indeterminate"><i /></div>)}
              </div>
              {openInfo === s.id && <div className="step-info">{STAGES.find(x => x.id === s.id)?.about ?? ''}</div>}
              {failed && error && s.id === stages.find(x => x.status === 'failed' || x.status === 'cancelled' || x.status === 'incomplete')?.id && (
                <div className="step-error">{error}</div>
              )}
              {openLog?.stage === s.id && <pre className="log">{openLog.text || 'Log is empty.'}</pre>}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
