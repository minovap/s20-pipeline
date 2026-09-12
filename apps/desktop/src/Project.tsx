// One project: scans on the left, the selected run (or a new run) on the right.
import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {AlertTriangle, ChevronLeft, FolderOpen, HardDrive, Play, Plus, Settings2, Square} from 'lucide-react';
import {api, errorText, pickCloud, pickFolder} from './api';
import {basename, bytes, count, duration, roughRange, when} from './format';
import {Pipeline} from './Pipeline';
import {statusWord} from './Projects';
import {ConfirmDialog, Modal, NameDialog, Segmented, Spinner, Toggle, useContextMenu} from './ui';
import {DEFAULT_OPTIONS, estimatedKeyframeCount, estimatedMatchingSpeedup, onExternalDrive, stagesFor} from './types';
import type {Input, Options, PhotoMatching, Project, Run, Settings, StageState} from './types';
import type {LiveRun} from './main';

type Selection = {kind: 'new'} | {kind: 'run'; path: string};

export function ProjectScreen(props: {
  path: string; settings: Settings; live: LiveRun | null; reloadKey: number;
  onBack: () => void; onSettings: () => void; onError: (m: string) => void;
  onStart: (project: Project, input: Input, options: Options) => Promise<void>;
  onResume: (project: Project, run: Run, overrides?: Partial<Options>) => Promise<void>;
  onCancel: () => void;
  onOpenViewer: (focus?: string) => void;
}) {
  const {path, settings, live, reloadKey, onError} = props;
  const [project, setProject] = useState<Project | null>(null);
  const [selected, setSelected] = useState<Selection | null>(() => (new URLSearchParams(location.search).get('select') === 'new' ? {kind: 'new'} : null));
  const [available, setAvailable] = useState<Record<string, boolean>>({});
  const [dialog, setDialog] = useState<React.ReactNode>(null);
  const [adding, setAdding] = useState(false);
  const {openMenu, menu} = useContextMenu();

  const load = useCallback(async () => {
    try {
      const p = await api.openProject(path);
      setProject(p);
      const checks = await Promise.all(p.inputs.map(async i => [i.path, await api.inputAvailable(i.path)] as const));
      setAvailable(Object.fromEntries(checks));
    } catch (e) { onError(errorText(e)); props.onBack(); }
  }, [path, onError]);
  useEffect(() => { void load(); }, [load, reloadKey]);

  const liveHere = live && live.project === path ? live : null;
  const busy = !!liveHere && (liveHere.status === 'running' || liveHere.status === 'starting');

  // Runs on disk, with the live run (which may not have a folder yet) on top.
  const runs = useMemo<Run[]>(() => {
    if (!project) return [];
    const list = [...project.runs];
    if (liveHere && !list.some(r => r.path === liveHere.job.output)) {
      list.unshift({path: liveHere.job.output, name: basename(liveHere.job.output), status: liveHere.status, stage: null, error: liveHere.error,
        started: liveHere.startedAt / 1000, finished: liveHere.finishedAt ? liveHere.finishedAt / 1000 : null, capture: liveHere.job.capture,
        options: liveHere.job, stages: [], result: null, result_points: null});
    }
    return list;
  }, [project, liveHere]);

  useEffect(() => {
    if (!project || selected) return;
    if (liveHere) setSelected({kind: 'run', path: liveHere.job.output});
    else if (runs.length) setSelected({kind: 'run', path: runs[0].path});
    else setSelected({kind: 'new'});
  }, [project, runs, liveHere, selected]);
  // Jump to a run when it starts.
  useEffect(() => { if (liveHere?.status === 'starting') setSelected({kind: 'run', path: liveHere.job.output}); }, [liveHere?.job.output, liveHere?.status]);

  async function addInput() {
    const p = await pickFolder('Choose a raw S20 scan folder');
    if (!p || !project) return;
    setAdding(true);
    try {
      const next = await api.addInput(project.path, p);
      setProject(next);
      setAvailable(a => ({...a, [p]: true}));
      const input = next.inputs.find(i => i.path === p) ?? next.inputs[next.inputs.length - 1];
      if (input && onExternalDrive(input.path)) askCopy(next, input);
    }
    catch (e) { onError(errorText(e)); }
    finally { setAdding(false); }
  }
  function askCopy(current: Project, input: Input) {
    const choose = async (copy: boolean) => {
      setDialog(null);
      try { setProject(await api.writeProject(current.path, {inputs: current.inputs.map(i => (i.path === input.path ? {...i, copy} : i))})); } catch (e) { onError(errorText(e)); }
    };
    setDialog(
      <Modal title="Scan is on an external drive" onClose={() => choose(false)} width={480}>
        <p className="note">Reading {bytes(input.capture.bag_bytes)} straight from a card or external drive makes every run slower. The scan can be copied to this Mac first as part of each run and removed again when the run finishes.</p>
        <footer><button onClick={() => choose(false)}>Read in place</button><button className="primary" onClick={() => choose(true)}>Copy before processing</button></footer>
      </Modal>
    );
  }
  async function setCopy(input: Input, copy: boolean) {
    if (!project) return;
    try { setProject(await api.writeProject(project.path, {inputs: project.inputs.map(i => (i.path === input.path ? {...i, copy} : i))})); } catch (e) { onError(errorText(e)); }
  }
  async function removeInput(input: Input) {
    if (!project) return;
    setDialog(<ConfirmDialog title="Remove scan" body={<>Remove <b>{input.name}</b> from this project? The folder itself is not touched.</>} confirm="Remove" danger
      onCancel={() => setDialog(null)} onConfirm={async () => { setDialog(null); try { setProject(await api.writeProject(project.path, {inputs: project.inputs.filter(i => i.path !== input.path)})); } catch (e) { onError(errorText(e)); } }} />);
  }
  async function importCloud() {
    const p = await pickCloud('Choose a LAS or PLY point cloud');
    if (!p || !project) return;
    if (project.clouds.some(c => c.path === p)) { props.onOpenViewer(p); return; }
    try { setProject(await api.writeProject(project.path, {clouds: [...project.clouds, {path: p, name: basename(p).replace(/\.(las|ply)$/i, ''), added: Date.now() / 1000}]})); }
    catch (e) { onError(errorText(e)); }
  }
  function rename() {
    if (!project) return;
    setDialog(<NameDialog title="Rename project" defaultValue={project.name} onCancel={() => setDialog(null)}
      onSubmit={async name => { setDialog(null); try { setProject(await api.writeProject(project.path, {name})); } catch (e) { onError(errorText(e)); } }} />);
  }
  function deleteRun(run: Run) {
    setDialog(<ConfirmDialog title="Delete run" body={<>Delete the run from {when(run.started)} and everything it produced? This cannot be undone.</>} confirm="Delete" danger
      onCancel={() => setDialog(null)} onConfirm={async () => { setDialog(null); try { await api.deleteRun(run.path); setSelected(null); await load(); } catch (e) { onError(errorText(e)); } }} />);
  }

  if (!project) return <div className="center"><Spinner /></div>;
  const selectedRun = selected?.kind === 'run' ? runs.find(r => r.path === selected.path) ?? null : null;

  return (
    <div className="screen project">
      <header className="bar">
        <div className="row">
          <button className="icon" onClick={props.onBack} aria-label="All projects" title="All projects"><ChevronLeft size={18} /></button>
          <h1 onContextMenu={e => openMenu(e, [{label: 'Rename', onClick: rename}, {label: 'Show in Finder', onClick: () => api.reveal(project.path)}])} title="Right-click to rename">{project.name}</h1>
        </div>
        <div className="row">
          <button onClick={() => props.onOpenViewer()} disabled={!runs.some(r => r.result) && !project.exports.length && !project.clouds.length}>Open viewer</button>
          <button className="icon" onClick={props.onSettings} aria-label="Settings" title="Settings"><Settings2 size={17} /></button>
        </div>
      </header>
      <div className="columns">
        <aside className="rail">
          <section>
            <h2>Scans</h2>
            <ul className="rows compact">
              {project.inputs.map(i => (
                <li key={i.path}>
                  <div className="rowbutton static" onContextMenu={e => openMenu(e, [
                    {label: i.copy ? 'Read in place instead' : 'Copy to this Mac before processing', onClick: () => setCopy(i, !i.copy), disabled: busy},
                    {label: 'Show in Finder', onClick: () => api.reveal(i.path), disabled: available[i.path] === false},
                    {label: 'Remove from project', onClick: () => removeInput(i), danger: true, disabled: busy}])}>
                    <strong>{i.name}{available[i.path] === false && <span className="warn" title="Folder not found. Mount the SD card or drive."><AlertTriangle size={12} /></span>}{i.copy && <span className="muted" title="Copied to this Mac before each run"><HardDrive size={12} /></span>}</strong>
                    <span className="muted">{duration(i.capture.bag_duration_s)} scan, {i.capture.lidar_frames.toLocaleString()} frames, {i.capture.photos} photos, {bytes(i.capture.bag_bytes)}</span>
                  </div>
                </li>
              ))}
            </ul>
            <button className="add" onClick={addInput} disabled={adding || !settings.engine_ready}>{adding ? <Spinner size={12} /> : <Plus size={14} />}{adding ? 'Reading scan' : 'Add scan folder'}</button>
          </section>
          <section>
            <h2>Runs</h2>
            <ul className="rows compact">
              {runs.map(r => {
                const isLive = liveHere?.job.output === r.path;
                const status = isLive ? liveHere!.status : r.status;
                const started = isLive ? liveHere!.startedAt / 1000 : r.started;
                const finished = isLive ? (liveHere!.finishedAt ? liveHere!.finishedAt / 1000 : null) : r.finished;
                return (
                  <li key={r.path}>
                    <button className={'rowbutton' + (selected?.kind === 'run' && selected.path === r.path ? ' selected' : '')} onClick={() => setSelected({kind: 'run', path: r.path})}
                      onContextMenu={e => openMenu(e, [{label: 'Show in Finder', onClick: () => api.reveal(r.path)}, {label: 'Delete run', danger: true, disabled: isLive && busy, onClick: () => deleteRun(r)}])}>
                      <strong>{when(started) || r.name}</strong>
                      <span className={'status ' + status}>{status === 'running' || status === 'starting' ? <><Spinner size={11} /> {statusWord(status)}</> : statusWord(status)}{finished && started ? `, ${duration(finished - started)}` : ''}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
            <button className={'add' + (selected?.kind === 'new' ? ' selected' : '')} onClick={() => setSelected({kind: 'new'})}><Plus size={14} />New run</button>
          </section>
          <section>
            <h2>Point clouds</h2>
            <ul className="rows compact">
              {runs.filter(r => r.result).map(r => (
                <li key={r.path}><button className="rowbutton" onClick={() => props.onOpenViewer(r.result!)} onContextMenu={e => openMenu(e, [{label: 'Show in Finder', onClick: () => api.reveal(r.result!)}])}>
                  <strong>{when(r.started)}</strong><span className="muted">{r.result_points ? `${count(r.result_points)} points` : basename(r.result!)}</span></button></li>
              ))}
              {project.exports.map(x => (
                <li key={x.path}><button className="rowbutton" onClick={() => props.onOpenViewer(x.path)} onContextMenu={e => openMenu(e, [{label: 'Show in Finder', onClick: () => api.reveal(x.path)}])}>
                  <strong>{x.name}</strong><span className="muted">Export, {bytes(x.bytes)}</span></button></li>
              ))}
              {project.clouds.map(c => (
                <li key={c.path}><button className="rowbutton" onClick={() => props.onOpenViewer(c.path)} onContextMenu={e => openMenu(e, [{label: 'Show in Finder', onClick: () => api.reveal(c.path)}, {label: 'Remove from project', danger: true, onClick: async () => { try { setProject(await api.writeProject(project.path, {clouds: project.clouds.filter(x => x.path !== c.path)})); } catch (e) { onError(errorText(e)); } }}])}>
                  <strong>{c.name}</strong><span className="muted">Imported</span></button></li>
              ))}
            </ul>
            <button className="add" onClick={importCloud}><FolderOpen size={14} />Import point cloud</button>
          </section>
        </aside>
        <main className="content">
          {selected?.kind === 'new' && (
            <NewRun project={project} settings={settings} busy={busy} available={available} onAddInput={addInput}
              onStart={(input, options) => props.onStart(project, input, options)} />
          )}
          {selectedRun && (
            <RunDetail run={selectedRun} live={liveHere?.job.output === selectedRun.path ? liveHere : null} project={project} busy={busy}
              onCancel={props.onCancel} onResume={overrides => props.onResume(project, selectedRun, overrides)} onOpenViewer={props.onOpenViewer} onDelete={() => deleteRun(selectedRun)} />
          )}
        </main>
      </div>
      {dialog}
      {menu}
    </div>
  );
}

/** Turn pipeline error text into something a person can act on. */
export function explainError(message: string | null | undefined, memoryGb?: number): string {
  if (!message) return '';
  if (/RSS budget/i.test(message)) return `This step needed more memory than the run's limit${memoryGb ? ` of ${memoryGb} GB` : ''}. Raise the memory limit and resume; the finished steps are kept.`;
  if (/Broken pipe/i.test(message)) return 'The app closed or restarted while processing was running. Resume to continue from the last finished step.';
  if (/Resume requires identical/i.test(message)) return 'The scan, the options or the pipeline code changed since this run started, so it cannot be resumed. Start a new run.';
  if (/Another processing job/i.test(message)) return 'Another run is already in progress. Wait for it to finish or cancel it.';
  return message.replace(/^s20: /, '');
}

function NewRun({project, settings, busy, available, onAddInput, onStart}:
  {project: Project; settings: Settings; busy: boolean; available: Record<string, boolean>; onAddInput: () => void; onStart: (input: Input, options: Options) => Promise<void>}) {
  const key = `options:${project.path}`;
  const [options, setOptions] = useState<Options>(() => { try {
    const saved = {...DEFAULT_OPTIONS, ...JSON.parse(localStorage.getItem(key) ?? '{}')};
    return {...saved, keyframe_percent: Math.max(10, Math.min(100, Math.round(saved.keyframe_percent / 10) * 10))};
  } catch { return DEFAULT_OPTIONS; } });
  useEffect(() => { localStorage.setItem(key, JSON.stringify(options)); }, [key, options]);
  // First time on this Mac: allow the pipeline about two thirds of installed memory.
  useEffect(() => {
    if (localStorage.getItem(key)) return;
    api.hardware().then(h => setOptions(o => ({...o, memory_gb: Math.max(8, Math.round((h.memory_bytes / 1e9) * 0.65))}))).catch(() => {});
  }, [key]);
  const [inputPath, setInputPath] = useState(project.inputs[project.inputs.length - 1]?.path ?? '');
  const input = project.inputs.find(i => i.path === inputPath) ?? project.inputs[project.inputs.length - 1];
  const [advanced, setAdvanced] = useState(false);
  const [starting, setStarting] = useState(false);
  const set = (patch: Partial<Options>) => setOptions(o => ({...o, ...patch}));
  const steps = stagesFor({...options, copy: !!input?.copy}).length;
  const estimate = input?.estimate?.range_seconds ? roughRange(input.estimate.range_seconds) : '';
  const canStart = !!input && available[input.path] !== false && settings.engine_ready && !busy && !starting;

  if (!project.inputs.length) {
    return <div className="empty"><h2>Add a scan to begin</h2><p>Choose the folder from the S20 that holds the recording. The folder is only read, never changed.</p><button className="primary" onClick={onAddInput} disabled={!settings.engine_ready}><Plus size={15} />Add scan folder</button></div>;
  }
  return (
    <div className="panel setup">
      <h2>New run</h2>
      {project.inputs.length > 1 && (
        <label className="field"><span>Scan</span>
          <select value={input?.path} onChange={e => setInputPath(e.target.value)}>{project.inputs.map(i => <option key={i.path} value={i.path}>{i.name}</option>)}</select>
        </label>
      )}
      <div className="field"><span>Result</span>
        <Segmented value={options.color ? 'color' : 'geometry'} onChange={v => set({color: v === 'color'})} options={[{value: 'color', label: 'Colored point cloud'}, {value: 'geometry', label: 'Geometry only'}]} />
      </div>
      <Toggle label="Remove people" hint="Masks people in the photos so they do not color the cloud" checked={options.mask === 'person'} disabled={!options.color} onChange={v => set({mask: v ? 'person' : 'off'})} />
      <div className="field"><span>Exposure</span>
        <Segmented value={options.exposure} disabled={!options.color} onChange={v => set({exposure: v})} options={[{value: 'local', label: 'Local'}, {value: 'global', label: 'Global'}, {value: 'off', label: 'Off'}]} />
        <small>{options.exposure === 'local' ? 'Evens out lighting within each photo and between photos.' : options.exposure === 'global' ? 'Evens out lighting between photos only.' : 'Photos are blended as recorded.'}</small>
      </div>
      {options.color && <div className="field"><span>Colorize pointcloud</span>
        <div className="colorize-settings">
          <div className="field"><span>Photo matching</span>
            <Segmented<PhotoMatching> value={options.photo_matching} onChange={v => set({photo_matching: v})} options={[{value: 'exact', label: 'Exact'}, {value: 'keyframes', label: 'Fast'}]} />
            <small>{options.photo_matching === 'exact' ? 'Checks every photo for the best coverage.' : 'Recommended for larger scans. Faster matching, with possible color gaps.'}</small>
          </div>
          {options.photo_matching === 'keyframes' && <div className="field">
            <label htmlFor="keyframe-percent">Photos to use: {options.keyframe_percent}%</label>
            <input id="keyframe-percent" type="range" min={10} max={100} step={10} value={options.keyframe_percent} onChange={e => set({keyframe_percent: Number(e.target.value)})} />
            <small>About {estimatedKeyframeCount(input?.capture.photos ?? 0, options.keyframe_percent)} of {input?.capture.photos ?? 0} recorded photos. {options.keyframe_percent === 100 ? 'Use Exact for all photos.' : input?.capture.photos ? `Estimated matching speed: ~${estimatedMatchingSpeedup(input.capture.photos, options.keyframe_percent).toFixed(1)}× faster (short-scan model).` : 'Matching speed estimate unavailable.'}</small>
          </div>}
        </div>
      </div>}
      <div className="field"><span>Performance</span>
        <Segmented value={options.resources} onChange={v => set({resources: v})} options={[{value: 'interactive', label: 'Light'}, {value: 'balanced', label: 'Balanced'}, {value: 'throughput', label: 'Max'}]} />
        <small>{options.resources === 'throughput' ? 'Uses every CPU core. The Mac will feel slow while it runs.' : options.resources === 'interactive' ? 'Leaves most of the CPU free for other work.' : 'Up to 8 cores.'}</small>
      </div>
      <button className="link" onClick={() => setAdvanced(a => !a)} aria-expanded={advanced}>{advanced ? 'Hide advanced' : 'Advanced'}</button>
      {advanced && (
        <div className="advanced">
          <label className="field inline"><span>Memory limit<small style={{display: 'block'}}>A step that needs more is stopped so the Mac stays usable.</small></span><span className="unit"><input type="number" min={1} max={1024} value={options.memory_gb} onChange={e => set({memory_gb: Math.max(1, Math.min(1024, +e.target.value || 1))})} /> GB</span></label>
          <Toggle label="Refine poses" hint="Second pass that tightens the scan trajectory. Recommended." checked={options.pose_refinement} onChange={v => set({pose_refinement: v})} />
        </div>
      )}
      <div className="start">
        <button className="primary large" disabled={!canStart} onClick={async () => { setStarting(true); try { await onStart(input!, options); } finally { setStarting(false); } }}><Play size={16} />Process</button>
        <span className="muted">{steps} steps{estimate ? `, usually ${estimate}` : ''}{input?.copy ? '. The scan is copied to this Mac first.' : onExternalDrive(input?.path ?? '') ? '. Reads from the external drive.' : ''}</span>
        {!settings.engine_ready && <span className="warn-text">The processing engine is not set up. Open Settings.</span>}
        {input && available[input.path] === false && <span className="warn-text">The scan folder is not mounted.</span>}
        {busy && <span className="muted">Another run is in progress.</span>}
      </div>
    </div>
  );
}

function RunDetail({run, live, project, busy, onCancel, onResume, onOpenViewer, onDelete}:
  {run: Run; live: LiveRun | null; project: Project; busy: boolean; onCancel: () => void; onResume: (overrides?: Partial<Options>) => void; onOpenViewer: (focus: string) => void; onDelete: () => void}) {
  const status = live ? live.status : run.status;
  const rawError = live?.error ?? run.error;
  const memoryError = /RSS budget/i.test(rawError ?? '');
  const [memory, setMemory] = useState<number>(Math.round((run.options.memory_gb || 16) * 2));
  const input = project.inputs.find(i => i.path === run.capture);
  const stages = useMemo<StageState[]>(() => {
    const forecast = stagesFor(run.options).map(s => s.id);
    if (live) {
      const ids = [...forecast];
      for (const id of live.order) if (!ids.includes(id)) ids.push(id);
      return ids.map(id => live.stages[id] ?? {id, status: 'pending'});
    }
    const seen = new Map(run.stages.map(s => [s.id, s]));
    const ids = [...forecast];
    for (const s of run.stages) if (!ids.includes(s.id)) ids.push(s.id);
    return ids.map(id => {
      const s = seen.get(id);
      if (s?.status === 'complete') return {id, status: 'complete', wall_s: s.wall_s};
      if (s) return {id, status: run.status === 'cancelled' ? 'cancelled' : run.status === 'failed' ? 'failed' : 'incomplete'};
      return {id, status: 'pending'};
    });
  }, [run, live]);
  const started = live ? live.startedAt : run.started ? run.started * 1000 : null;
  const finished = live ? live.finishedAt : run.finished ? run.finished * 1000 : null;
  const resumable = !busy && (status === 'cancelled' || status === 'failed' || status === 'interrupted') && stages.some(s => s.status === 'complete');
  const result = live?.status === 'completed' ? (run.result ?? `${run.path}/${run.options.color ? 'export/colorized.las' : 'geometry/filtered.ply'}`) : run.result;

  return (
    <div className="panel run">
      <div className="run-head">
        <div>
          <h2>{when(started ? started / 1000 : null) || run.name}</h2>
          <span className="muted">{input?.name ?? basename(run.capture)}, {run.options.color ? 'colored' : 'geometry only'}{run.options.color && run.options.mask === 'person' ? ', people removed' : ''}{run.options.color && run.options.photo_matching === 'keyframes' ? `, fast photo matching (${run.options.keyframe_percent ?? 30}%)` : ''}, {run.options.resources === 'throughput' ? 'max' : run.options.resources === 'interactive' ? 'light' : 'balanced'} performance{run.options.copy ? ', copied first' : ''}</span>
        </div>
        <div className="row">
          {(status === 'running' || status === 'starting') && <button className="danger" onClick={onCancel}><Square size={13} />Cancel</button>}
          {resumable && memoryError && <label className="field inline resume-memory"><span>Memory limit</span><span className="unit"><input type="number" min={1} max={1024} value={memory} onChange={e => setMemory(Math.max(1, Math.min(1024, +e.target.value || 1)))} /> GB</span></label>}
          {resumable && <button className="primary" onClick={() => onResume(memoryError ? {memory_gb: memory} : undefined)}><Play size={14} />Resume</button>}
          {!busy && status !== 'running' && <button onClick={onDelete}>Delete</button>}
        </div>
      </div>
      {status === 'completed' && result && (
        <div className="result">
          <div><strong>Point cloud ready</strong><span className="muted">{run.result_points ? `${count(run.result_points)} points, ` : ''}{basename(result)}</span></div>
          <div className="row"><button onClick={() => api.reveal(result)}>Show in Finder</button><button className="primary" onClick={() => onOpenViewer(result)}>Open in viewer</button></div>
        </div>
      )}
      <Pipeline stages={stages} status={status} startedAt={started} finishedAt={finished} error={explainError(rawError, run.options.memory_gb)} cpu={live?.cpu} memory={live?.memory}
        onShowLog={stage => api.readStageLog(run.path, stage)} />
    </div>
  );
}
