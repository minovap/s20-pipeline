// App root: settings, screen routing and the single live processing run.
// Pipeline events are subscribed once here so a run keeps updating while the
// user is on another screen.
import React, {useCallback, useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {api, errorText, inTauri, listen, mock, pickFolder} from './api';
import {Projects} from './Projects';
import {ProjectScreen} from './Project';
import {Viewer} from './viewer/Viewer';
import {ErrorBar, Modal} from './ui';
import {runFolderName} from './format';
import {stagesFor} from './types';
import type {Input, Job, Options, PipelineEvent, Project, Run, RunStatus, Settings, StageState} from './types';
import './style.css';

export type LiveRun = {
  project: string; job: Job; status: RunStatus; order: string[]; stages: Record<string, StageState>;
  startedAt: number; finishedAt: number | null; error: string | null; cpu: number | null; memory: number | null; logs: string[];
};
type Screen = {kind: 'projects'} | {kind: 'project'; path: string} | {kind: 'viewer'; path: string; focus?: string};

/** Module-level copy of the live run so a remount (hot reload, error boundary) does not lose it. */
let liveSnapshot: LiveRun | null = null;

function App() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [screen, setScreen] = useState<Screen>(() => {
    if (mock) {
      const s = new URLSearchParams(location.search).get('screen');
      const path = '/Users/demo/Documents/S20 Projects/Garden';
      return s === 'viewer' ? {kind: 'viewer', path} : s === 'project' || s === 'running' ? {kind: 'project', path} : {kind: 'projects'};
    }
    const last = localStorage.getItem('lastProject');
    return last ? {kind: 'project', path: last} : {kind: 'projects'};
  });
  const [live, setLive] = useState<LiveRun | null>(() => liveSnapshot);
  const liveRef = useRef<LiveRun | null>(null);
  liveRef.current = live;
  liveSnapshot = live;
  const [error, setError] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  const refreshSettings = useCallback(async () => {
    try { setSettings(await api.settings()); } catch (e) { setError(errorText(e)); }
  }, []);
  useEffect(() => { if (inTauri || mock) void refreshSettings(); }, [refreshSettings]);

  useEffect(() => {
    if (!inTauri && !mock) return;
    let disposed = false;
    const offs: (() => void)[] = [];
    const sub = <T,>(name: string, fn: (data: T) => void) => {
      listen<T>(name, payload => { if (!disposed) fn(payload); }).then(off => (disposed ? off() : offs.push(off)));
    };
    const update = (fn: (run: LiveRun) => LiveRun) => setLive(current => (current ? fn(current) : current));
    sub<PipelineEvent>('pipeline-event', e => {
      if (e.run_id !== liveRef.current?.job.output) return;
      const t = Date.now();
      if (e.event === 'stage_started' && e.stage) {
        const id = e.stage;
        update(r => ({...r, status: 'running', order: r.order.includes(id) ? r.order : [...r.order, id], stages: {...r.stages, [id]: {id, status: 'running', startedAt: t}}}));
      } else if (e.event === 'stage_cached' && e.stage) {
        const id = e.stage;
        update(r => ({...r, status: 'running', order: r.order.includes(id) ? r.order : [...r.order, id], stages: {...r.stages, [id]: {id, status: 'cached'}}}));
      } else if (e.event === 'progress' && e.stage) {
        const id = e.stage;
        update(r => ({...r, stages: {...r.stages, [id]: {...r.stages[id], id, status: 'running', done: e.done, total: e.total, unit: e.unit}}}));
      } else if (e.event === 'stage_completed' && e.stage) {
        const id = e.stage;
        update(r => ({...r, stages: {...r.stages, [id]: {id, status: 'complete', wall_s: e.wall_s ?? (r.stages[id]?.startedAt ? (t - r.stages[id].startedAt!) / 1000 : null)}}}));
      } else if (e.event === 'resources') {
        update(r => ({...r, cpu: e.cpu_core_equivalents ?? null, memory: e.rss_bytes ?? null}));
      } else if (e.event === 'failed' || e.event === 'cancelled') {
        const kind = e.event;
        update(r => ({...r, status: kind, finishedAt: t, error: e.message ?? r.error, stages: e.stage ? {...r.stages, [e.stage]: {...r.stages[e.stage], id: e.stage, status: kind}} : r.stages}));
      } else if (e.event === 'completed') {
        update(r => ({...r, status: 'completed', finishedAt: t}));
      }
    });
    sub<{run_id: string; message: string}>('job-log', e => {
      if (e.run_id !== liveRef.current?.job.output) return;
      update(r => ({...r, logs: [...r.logs.slice(-199), e.message]}));
    });
    sub<{run_id: string; code: number}>('job-exit', e => {
      if (e.run_id !== liveRef.current?.job.output) return;
      update(r => {
        const status: RunStatus = r.status === 'completed' || r.status === 'failed' || r.status === 'cancelled' ? r.status : e.code === 0 ? 'completed' : e.code === 130 ? 'cancelled' : 'failed';
        // A failure before the first stage (bad inputs, missing binary) has no stage event; keep the last stderr line as the reason.
        const error = r.error ?? (status === 'failed' ? (r.logs.filter(l => l.trim()).pop() ?? 'Processing failed before the first step.') : null);
        return {...r, status, finishedAt: r.finishedAt ?? Date.now(), error};
      });
      setReloadKey(k => k + 1);
    });
    sub<string>('close-blocked', setError);
    return () => { disposed = true; offs.forEach(off => off()); };
  }, []);

  async function startRun(project: Project, input: Input, options: Options) {
    const job: Job = {...options, copy: !!input.copy, capture: input.path, output: `${project.path}/runs/${runFolderName()}`, resume: false};
    await launch(project.path, job);
  }
  async function resumeRun(project: Project, run: Run, overrides: Partial<Options> = {}) {
    const job: Job = {...run.options, ...overrides, copy: !!run.options.copy, capture: run.capture, output: run.path, resume: true};
    await launch(project.path, job, run);
  }
  async function launch(projectPath: string, job: Job, previous?: Run) {
    const forecast = stagesFor(job).map(s => s.id);
    const stages: Record<string, StageState> = {};
    for (const id of forecast) stages[id] = {id, status: 'pending'};
    if (previous) for (const s of previous.stages) if (s.status === 'complete') stages[s.id] = {id: s.id, status: 'pending', wall_s: s.wall_s};
    setLive({project: projectPath, job, status: 'starting', order: [], stages, startedAt: Date.now(), finishedAt: null, error: null, cpu: null, memory: null, logs: []});
    setError('');
    try { await api.startJob(job); }
    catch (e) { setLive(r => (r ? {...r, status: 'failed', finishedAt: Date.now(), error: errorText(e)} : r)); }
  }
  async function cancelRun() {
    try { await api.cancelJob(); } catch (e) { setError(errorText(e)); }
  }

  const openProject = (path: string) => { localStorage.setItem('lastProject', path); setScreen({kind: 'project', path}); };
  const openProjects = () => { localStorage.removeItem('lastProject'); setScreen({kind: 'projects'}); };

  // Review mode: ?mock&screen=running starts a simulated run on load.
  useEffect(() => {
    if (!mock || new URLSearchParams(location.search).get('screen') !== 'running' || !settings) return;
    void api.openProject('').then(p => startRun(p, p.inputs[0], {resources: 'throughput', memory_gb: 16, color: true, mask: 'person', exposure: 'local', pose_refinement: true}));
  }, [settings]);

  if (!inTauri && !mock) return <div className="center"><p>Open S20 Studio through the desktop app.</p></div>;
  if (!settings) return <div className="center"><p>Starting</p></div>;

  return (
    <>
      {screen.kind === 'projects' && (
        <Projects settings={settings} live={live} onOpen={openProject} onSettings={() => setShowSettings(true)} onError={setError} />
      )}
      {screen.kind === 'project' && (
        <ProjectScreen key={screen.path} path={screen.path} settings={settings} live={live} reloadKey={reloadKey}
          onBack={openProjects} onSettings={() => setShowSettings(true)} onError={setError}
          onStart={startRun} onResume={resumeRun} onCancel={cancelRun}
          onOpenViewer={(focus?: string) => setScreen({kind: 'viewer', path: screen.path, focus})} />
      )}
      {screen.kind === 'viewer' && (
        <Viewer key={screen.path} path={screen.path} focus={screen.focus} onBack={() => setScreen({kind: 'project', path: screen.path})} onError={setError} />
      )}
      <div className="toasts"><ErrorBar message={error} onClose={() => setError('')} /></div>
      {showSettings && <SettingsDialog settings={settings} busy={!!live && (live.status === 'running' || live.status === 'starting')} onChange={refreshSettings} onClose={() => setShowSettings(false)} onError={setError} />}
    </>
  );
}

function SettingsDialog({settings, busy, onChange, onClose, onError}: {settings: Settings; busy: boolean; onChange: () => void; onClose: () => void; onError: (m: string) => void}) {
  async function chooseEngine() {
    const p = await pickFolder('Choose the s20-pipeline folder');
    if (!p) return;
    try { await api.configure(p); onChange(); } catch (e) { onError(errorText(e)); }
  }
  async function chooseProjects() {
    const p = await pickFolder('Choose where projects are stored');
    if (!p) return;
    try { await api.setProjectsRoot(p); onChange(); } catch (e) { onError(errorText(e)); }
  }
  return (
    <Modal title="Settings" onClose={onClose} width={520}>
      <div className="setting">
        <span>Projects folder</span>
        <code title={settings.projects_root}>{settings.projects_root}</code>
        <div className="row"><button onClick={chooseProjects}>Change</button><button onClick={() => api.reveal(settings.projects_root)}>Show in Finder</button></div>
      </div>
      <div className="setting">
        <span>Processing engine</span>
        <code title={settings.engine_root}>{settings.engine_root}</code>
        <p className="note">{settings.engine_ready ? 'Python environment and native binaries found.' : 'Not usable yet: the folder needs a built .venv and build/s20_geometry. See the pipeline README.'}</p>
        <div className="row"><button onClick={chooseEngine} disabled={busy}>Change</button></div>
      </div>
    </Modal>
  );
}

/** Last line of defence: show the error instead of a blank window. */
class Boundary extends React.Component<{children: React.ReactNode}, {error: string}> {
  state = {error: ''};
  static getDerivedStateFromError(e: unknown) { return {error: e instanceof Error ? `${e.message}\n${e.stack ?? ''}` : String(e)}; }
  render() {
    if (!this.state.error) return this.props.children;
    return <div className="center"><div className="crash"><h2>Something went wrong</h2><pre>{this.state.error}</pre><button className="primary" onClick={() => location.reload()}>Reload</button></div></div>;
  }
}

createRoot(document.getElementById('root')!).render(<Boundary><App /></Boundary>);
