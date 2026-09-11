// Typed wrappers around the Rust commands. Everything that touches disk or
// spawns a process lives behind one of these.
import {invoke, isTauri} from '@tauri-apps/api/core';
import {listen as tauriListen} from '@tauri-apps/api/event';
import {open} from '@tauri-apps/plugin-dialog';
import type {Box, Hardware, Job, Preview, Project, ProjectSummary, Settings} from './types';

export const inTauri = isTauri();
/** Browser review mode: `?mock` on the Vite dev server swaps in fake data. */
export const mock = !inTauri && typeof location !== 'undefined' && new URLSearchParams(location.search).has('mock');

const real = {
  settings: () => invoke<Settings>('settings'),
  configure: (path: string) => invoke<void>('configure', {path}),
  setProjectsRoot: (path: string) => invoke<void>('set_projects_root', {path}),
  hardware: () => invoke<Hardware>('hardware'),
  reveal: (path: string) => invoke<void>('reveal', {path}),

  listProjects: () => invoke<ProjectSummary[]>('list_projects'),
  createProject: (name: string) => invoke<Project>('create_project', {name}),
  openProject: (path: string) => invoke<Project>('open_project', {path}),
  writeProject: (path: string, project: Partial<Pick<Project, 'name' | 'inputs' | 'clouds' | 'slices'>>) => invoke<Project>('write_project', {path, project}),
  addInput: (project: string, capture: string) => invoke<Project>('add_input', {project, capture}),
  inputAvailable: (path: string) => invoke<boolean>('input_available', {path}),
  readStageLog: (run: string, stage: string) => invoke<string>('read_stage_log', {run, stage}),
  deleteRun: (run: string) => invoke<void>('delete_run', {run}),
  deleteExport: (path: string) => invoke<void>('delete_export', {path}),

  startJob: (options: Job) => invoke<void>('start_job', {options}),
  cancelJob: () => invoke<void>('cancel_job'),

  exportSlices: (spec: {output: string; sources: {path: string; boxes: Box[]}[]; allow_outside?: boolean}) => invoke<string>('export_slices', {spec}),
  cancelExport: () => invoke<void>('cancel_export'),

  loadPreview: (source: string, budget: number) => invoke<Preview>('load_preview', {source, budget}),
  readPreview: (key: string) => invoke<ArrayBuffer>('read_preview', {key}),
};
export const api: typeof real = mock ? (await import('./mock')).mockApi as unknown as typeof real : real;

/** Subscribe to a shell event. Returns an unsubscribe function once ready. */
export async function listen<T>(name: string, fn: (payload: T) => void): Promise<() => void> {
  if (mock) return (await import('./mock')).mockListen(name, fn as (p: unknown) => void);
  return tauriListen<T>(name, e => fn(e.payload));
}

export async function pickFolder(title: string): Promise<string | null> {
  if (mock) return '/Volumes/SD_CARD/Another-scan';
  const p = await open({directory: true, multiple: false, title});
  return typeof p === 'string' ? p : null;
}
export async function pickCloud(title: string): Promise<string | null> {
  if (mock) return '/Users/demo/Downloads/reference.las';
  const p = await open({multiple: false, title, filters: [{name: 'Point cloud', extensions: ['las', 'ply']}]});
  return typeof p === 'string' ? p : null;
}

export const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e)).replace(/^s20: /, '');
