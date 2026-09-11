// Shared types for the desktop app. Field names mirror the JSON written by
// the Rust shell and the Python pipeline so nothing is renamed in transit.

export type Hardware = {cpu_model: string; logical_cpu_cores: number; memory_bytes: number; available_memory_bytes: number; machine: string};

export type Capture = {
  capture: string; bag_bytes: number; bag_duration_s: number; photos: number; lidar_frames: number; imu_samples: number;
  device: {device_model: string; lidar_model: string; work_duration: number};
  cameras: Record<string, {width: number; height: number}>;
};
export type Estimate = {estimated_seconds: number | null; range_seconds: [number, number] | null; confidence: string};

export type Input = {path: string; name: string; added: number; capture: Capture; estimate?: Estimate | null};

export type Options = {resources: string; memory_gb: number; color: boolean; mask: string; exposure: string; pose_refinement: boolean};
export const DEFAULT_OPTIONS: Options = {resources: 'balanced', memory_gb: 16, color: true, mask: 'person', exposure: 'local', pose_refinement: true};

export type Job = Options & {capture: string; output: string; resume: boolean};

export type StageStatus = 'pending' | 'running' | 'complete' | 'cached' | 'failed' | 'cancelled' | 'incomplete' | 'skipped';
export type StageState = {id: string; status: StageStatus; done?: number; total?: number; wall_s?: number | null; startedAt?: number};

export type RunStatus = 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted' | 'starting' | 'unknown';
export type Run = {
  path: string; name: string; status: RunStatus; stage: string | null; error: string | null;
  started: number | null; finished: number | null; capture: string; options: Options;
  stages: {id: string; status: 'complete' | 'incomplete'; wall_s: number | null}[];
  result: string | null; result_points: number | null;
};

export type Box = [[number, number, number], [number, number, number]];
export type Slice = {id: string; name: string; source: string; parent: string | null; box: Box; created: number};
export type ImportedCloud = {path: string; name: string; added: number};
/** Rotation in degrees (roll about X, pitch about Y, yaw about Z; applied yaw·pitch·roll) about the cloud's preview origin, then a shift in metres. */
export type Orientation = {rotation: [number, number, number]; translation: [number, number, number]};
export type ExportFile = {path: string; name: string; bytes: number | null; modified: number | null};

export type ProjectSummary = {path: string; name: string; created: number; input_count: number; run_count: number; last_run: Run | null};
export type Project = {
  path: string; name: string; created: number; inputs: Input[]; clouds: ImportedCloud[]; slices: Slice[]; runs: Run[]; exports: ExportFile[];
  orientations: Record<string, Orientation>;
};

export type Settings = {engine_root: string; engine_ready: boolean; projects_root: string; running: boolean};

export type PipelineEvent = {
  event: string; stage?: string; done?: number; total?: number; wall_s?: number; rss_bytes?: number; cpu_core_equivalents?: number | null;
  system_available_memory_bytes?: number; message?: string; run_id: string; time_unix?: number;
};
export type ExportEvent = {id: string; name: string; event: 'progress' | 'completed' | 'failed' | 'cancelled'; done?: number; total?: number; file?: string; points?: number; message?: string};

export type Preview = {key: string; name: string; source: string; source_points: number; display_points: number; origin: number[]; bounds: number[][]; bytes: number};
export type Cloud = {info: Preview; data: Float32Array};

// Stage order matches s20_pipeline.runner.stage_names. Groups give the long
// list a shape; labels say what the step produces in plain words.
export const STAGES: {id: string; label: string; group: 'Geometry' | 'Color'}[] = [
  {id: 'decode', label: 'Decode LiDAR and IMU', group: 'Geometry'},
  {id: 'pack', label: 'Validate and pack frames', group: 'Geometry'},
  {id: 'tracking', label: 'Track motion and deskew', group: 'Geometry'},
  {id: 'pose_refinement', label: 'Refine poses', group: 'Geometry'},
  {id: 'registered', label: 'Register scan frames', group: 'Geometry'},
  {id: 'geometry', label: 'Filter geometry', group: 'Geometry'},
  {id: 'photos', label: 'Extract photos', group: 'Color'},
  {id: 'cameras', label: 'Calibrate cameras', group: 'Color'},
  {id: 'masks', label: 'Mask people', group: 'Color'},
  {id: 'candidates', label: 'Match photos to points', group: 'Color'},
  {id: 'global', label: 'Balance exposure between photos', group: 'Color'},
  {id: 'local', label: 'Balance exposure within photos', group: 'Color'},
  {id: 'blend', label: 'Blend colors', group: 'Color'},
  {id: 'export', label: 'Write point cloud', group: 'Color'},
];
export const stageLabel = (id: string) => STAGES.find(s => s.id === id)?.label ?? id;

export function stagesFor(job: Pick<Options, 'color' | 'pose_refinement' | 'mask' | 'exposure'>) {
  return STAGES.filter(({id, group}) => {
    if (id === 'pose_refinement' && !job.pose_refinement) return false;
    if (group === 'Color' && !job.color) return false;
    if (id === 'masks' && job.mask === 'off') return false;
    if ((id === 'global' || id === 'local') && job.exposure === 'off') return false;
    if (id === 'local' && job.exposure !== 'local') return false;
    return true;
  });
}
