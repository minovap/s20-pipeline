// Typed wrappers around the Rust commands. Everything that touches disk or
// spawns a process lives behind one of these.
import {convertFileSrc, invoke, isTauri} from '@tauri-apps/api/core';
import {listen as tauriListen} from '@tauri-apps/api/event';
import {open} from '@tauri-apps/plugin-dialog';
import type {Box, Cloud, CloudInfo, Hardware, Job, Preview, Project, ProjectSummary, Settings} from './types';

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
  writeProject: (path: string, project: Partial<Pick<Project, 'name' | 'inputs' | 'clouds' | 'slices' | 'orientations'>>) => invoke<Project>('write_project', {path, project}),
  addInput: (project: string, capture: string) => invoke<Project>('add_input', {project, capture}),
  inputAvailable: (path: string) => invoke<boolean>('input_available', {path}),
  readStageLog: (run: string, stage: string) => invoke<string>('read_stage_log', {run, stage}),
  deleteRun: (run: string) => invoke<void>('delete_run', {run}),
  deleteExport: (path: string) => invoke<void>('delete_export', {path}),
  deleteProject: (path: string) => invoke<void>('delete_project', {path}),
  tempCopies: () => invoke<{bytes: number; count: number; unreferenced_bytes: number; unreferenced: number}>('temp_copies'),
  cleanTempCopies: () => invoke<number>('clean_temp_copies'),

  startJob: (options: Job) => invoke<void>('start_job', {options}),
  cancelJob: () => invoke<void>('cancel_job'),

  exportSlices: (spec: {output: string; sources: {path: string; boxes?: Box[]; regions?: {box: Box; polygons: {vertices: [number, number][]; mode: 'inside' | 'ring'; expand: number}[]}[]; transform?: {rotation: number[]; origin: number[]; translation: number[]} | null}[]; allow_outside?: boolean}) => invoke<string>('export_slices', {spec}),
  cancelExport: () => invoke<void>('cancel_export'),

  loadPreview: (source: string, budget: number) => invoke<Preview>('load_preview', {source, budget}),
  readPreview: (key: string) => invoke<ArrayBuffer>('read_preview', {key}),
  cloudInfo: (source: string) => invoke<CloudInfo>('cloud_info', {source}),
  /** Sample a cloud and read its two preview files. */
  async loadCloud(source: string, budget: number): Promise<Cloud> {
    const info = await this.loadPreview(source, budget);
    const [positions, colors] = await Promise.all([readFile(info.file, info.key, info.bytes), readFile(info.colors, `${info.key}:colors`, info.color_bytes)]);
    return {info, positions: new Float32Array(positions), colors: new Uint8Array(colors)};
  },
};
export const api: typeof real = mock ? (await import('./mock')).mockApi as unknown as typeof real : real;

/** Large files stream through the asset protocol; IPC is the fallback. */
async function readFile(path: string, key: string, expected: number): Promise<ArrayBuffer> {
  let raw: ArrayBuffer | null = null;
  try {
    const response = await fetch(convertFileSrc(path));
    if (response.ok) raw = await response.arrayBuffer();
  } catch { raw = null; }
  if (!raw || raw.byteLength !== expected) raw = await invoke<ArrayBuffer>('read_preview', {key});
  if (raw.byteLength !== expected) throw new Error('Preview data was incomplete. Try again.');
  return raw;
}

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
