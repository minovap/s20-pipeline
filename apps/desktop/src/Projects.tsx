// Home screen: the list of projects.
import React, {useCallback, useEffect, useState} from 'react';
import {Plus, Settings2} from 'lucide-react';
import {api, errorText} from './api';
import {when} from './format';
import {NameDialog, Spinner, useContextMenu} from './ui';
import type {LiveRun} from './main';
import type {ProjectSummary, Run, Settings} from './types';

export function statusWord(status: Run['status'] | 'starting'): string {
  return {running: 'Processing', starting: 'Starting', completed: 'Done', failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted', unknown: ''}[status] ?? '';
}

export function Projects({settings, live, onOpen, onSettings, onError}:
  {settings: Settings; live: LiveRun | null; onOpen: (path: string) => void; onSettings: () => void; onError: (m: string) => void}) {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [creating, setCreating] = useState(false);
  const {openMenu, menu} = useContextMenu();

  const load = useCallback(async () => {
    try { setProjects(await api.listProjects()); } catch (e) { onError(errorText(e)); setProjects([]); }
  }, [onError]);
  useEffect(() => { void load(); }, [load, settings.projects_root]);

  async function create(name: string) {
    setCreating(false);
    try { const p = await api.createProject(name); onOpen(p.path); } catch (e) { onError(errorText(e)); }
  }

  return (
    <div className="screen projects">
      <header className="bar">
        <h1>Projects</h1>
        <div className="row">
          <button className="primary" onClick={() => setCreating(true)}><Plus size={15} />New project</button>
          <button className="icon" onClick={onSettings} aria-label="Settings" title="Settings"><Settings2 size={17} /></button>
        </div>
      </header>
      {!settings.engine_ready && (
        <div className="notice">Processing is not available yet. <button className="link" onClick={onSettings}>Choose the pipeline folder in Settings</button> to enable it. Viewing existing point clouds still works.</div>
      )}
      <main className="list-page">
        {projects === null ? <p className="muted">Loading</p> :
         projects.length === 0 ? (
          <div className="empty">
            <h2>No projects yet</h2>
            <p>A project holds the raw scan folders you add to it, every processing run and the point clouds they produce.</p>
            <button className="primary" onClick={() => setCreating(true)}><Plus size={15} />New project</button>
          </div>
         ) : (
          <ul className="rows">
            {projects.map(p => {
              const isLive = live?.project === p.path && (live.status === 'running' || live.status === 'starting');
              const last = p.last_run;
              return (
                <li key={p.path}>
                  <button className="rowbutton" onClick={() => onOpen(p.path)} onContextMenu={e => openMenu(e, [{label: 'Show in Finder', onClick: () => api.reveal(p.path)}])}>
                    <strong>{p.name}</strong>
                    <span className="muted">{p.input_count === 1 ? '1 scan' : `${p.input_count} scans`}, {p.run_count === 1 ? '1 run' : `${p.run_count} runs`}</span>
                    <span className={'status ' + (isLive ? 'running' : last?.status ?? '')}>
                      {isLive ? <><Spinner size={12} /> Processing</> : last ? `${statusWord(last.status)} ${when(last.started)}` : 'No runs yet'}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
         )}
      </main>
      {creating && <NameDialog title="New project" defaultValue="Garden" confirm="Create" onSubmit={create} onCancel={() => setCreating(false)} />}
      {menu}
    </div>
  );
}
