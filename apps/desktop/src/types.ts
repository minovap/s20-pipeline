// Shared types for the desktop app. Field names mirror the JSON written by
// the Rust shell and the Python pipeline so nothing is renamed in transit.

export type Hardware = {cpu_model: string; logical_cpu_cores: number; memory_bytes: number; available_memory_bytes: number; machine: string};

export type Capture = {
  capture: string; bag_bytes: number; bag_duration_s: number; photos: number; lidar_frames: number; imu_samples: number;
  device: {device_model: string; lidar_model: string; work_duration: number};
  cameras: Record<string, {width: number; height: number}>;
};
export type Estimate = {estimated_seconds: number | null; range_seconds: [number, number] | null; confidence: string};

export type Input = {path: string; name: string; added: number; capture: Capture; estimate?: Estimate | null; copy?: boolean};
/** External drives on macOS mount under /Volumes. */
export const onExternalDrive = (path: string) => path.startsWith('/Volumes/');

export type PhotoMatching = 'exact' | 'keyframes';
export type Options = {resources: string; memory_gb: number; color: boolean; mask: string; exposure: string; photo_matching: PhotoMatching; keyframe_percent: number; pose_refinement: boolean; copy?: boolean};
export const DEFAULT_OPTIONS: Options = {resources: 'balanced', memory_gb: 16, color: true, mask: 'person', exposure: 'local', photo_matching: 'exact', keyframe_percent: 30, pose_refinement: true};

export function estimatedKeyframeCount(photos: number, percent: number) {
  let selected = 0;
  for (let start = 0; start < photos; start += 62) {
    const window = Math.min(62, photos - start);
    let count = Math.min(window, Math.max(1, Math.round(window * percent / 100)));
    if (window >= 20 && count % 2 && count < window) count += 1;
    selected += count;
  }
  return selected;
}

/** Matching-stage estimate fitted to paired 62-photo runs, not whole-pipeline time. */
export function estimatedMatchingSpeedup(photos: number, percent: number) {
  if (photos <= 0) return 1;
  const selectedFraction = estimatedKeyframeCount(photos, percent) / photos;
  return 1 / (0.31 + 0.69 * selectedFraction);
}

export type Job = Options & {capture: string; output: string; resume: boolean};

export type StageStatus = 'pending' | 'waiting' | 'running' | 'complete' | 'cached' | 'failed' | 'cancelled' | 'incomplete' | 'skipped';
export type StageState = {
  id: string; status: StageStatus; done?: number; total?: number; unit?: string; phase?: string; wall_s?: number | null; startedAt?: number; waitingFor?: string[];
  /** Recent (time ms, done) pairs for the rate estimate. */
  samples?: [number, number][];
};
/** Size of the scan a run processes; drives time predictions from history. */
export type RunSize = {frames: number; photos: number};

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
  event: string; stage?: string; done?: number; total?: number; unit?: string; phase?: string; wall_s?: number; waiting_for?: string[]; rss_bytes?: number; cpu_core_equivalents?: number | null;
  system_available_memory_bytes?: number; message?: string; run_id: string; time_unix?: number;
};
export type ExportEvent = {id: string; name: string; event: 'progress' | 'completed' | 'failed' | 'cancelled'; done?: number; total?: number; file?: string; points?: number; message?: string};

export type Preview = {key: string; name: string; source: string; source_points: number; display_points: number; origin: number[]; bounds: number[][]; bytes: number};
export type Cloud = {info: Preview; data: Float32Array};

// Stage order matches s20_pipeline.runner.stage_names. Groups give the long
// list a shape; labels say what the step produces in plain words.
export const STAGES: {id: string; label: string; group: 'Prepare' | 'Geometry' | 'Color'; about: string}[] = [
  {id: 'copy', label: 'Copy scan to this Mac', group: 'Prepare', about: 'Copies the raw scan folder from the external drive to this Mac so every later step reads from the fast internal disk. The copy is removed when the run succeeds.'},
  {id: 'decode', label: 'Decode LiDAR and IMU', group: 'Geometry', about: 'Reads the raw recording and unpacks the LiDAR returns and the motion sensor samples into working files.'},
  {id: 'pack', label: 'Validate and pack frames', group: 'Geometry', about: 'Checks timestamps and calibration, then packs the frames into the compact format the native tracker reads.'},
  {id: 'tracking', label: 'Track motion and deskew', group: 'Geometry', about: 'Follows how the scanner moved, frame by frame, and straightens each sweep. The LiDAR keeps moving while it records, so without this every sweep would be smeared.'},
  {id: 'pose_refinement', label: 'Refine poses', group: 'Geometry', about: 'A second pass over the path that tightens the scanner positions wherever the same surface was seen more than once.'},
  {id: 'registered', label: 'Register scan frames', group: 'Geometry', about: 'Places every sweep into one shared coordinate frame using the final positions.'},
  {id: 'geometry', label: 'Filter geometry', group: 'Geometry', about: 'Merges the sweeps into one cloud on the GPU, removes stray points and estimates which way each surface faces.'},
  {id: 'photos', label: 'Extract photos', group: 'Color', about: 'Pulls the camera photos out of the recording as JPEG files.'},
  {id: 'masks', label: 'Mask people', group: 'Color', about: 'Finds people in the photos so they do not leave their colors on the cloud.'},
  {id: 'cameras', label: 'Calibrate cameras', group: 'Color', about: 'Works out where each photo was taken by matching its timestamp to the scanner path and the camera calibration.'},
  {id: 'candidates', label: 'Match photos to points', group: 'Color', about: 'For every point, picks the photos that actually see it, taking into account what is hidden behind other surfaces. Usually the longest step.'},
  {id: 'global', label: 'Balance exposure between photos', group: 'Color', about: 'Evens out brightness differences between photos so seams do not show where they meet.'},
  {id: 'local', label: 'Balance exposure within photos', group: 'Color', about: 'Evens out lighting differences inside each photo, such as sun and shade, before blending.'},
  {id: 'blend', label: 'Blend colors', group: 'Color', about: 'Combines the chosen photos into one color for every point.'},
  {id: 'export', label: 'Write point cloud', group: 'Color', about: 'Writes the finished colored point cloud as a LAS file in the run folder.'},
];
export const stageLabel = (id: string) => STAGES.find(s => s.id === id)?.label ?? id;

export function stagesFor(job: Pick<Options, 'color' | 'pose_refinement' | 'mask' | 'exposure' | 'copy'>) {
  return STAGES.filter(({id, group}) => {
    if (id === 'copy' && !job.copy) return false;
    if (id === 'pose_refinement' && !job.pose_refinement) return false;
    if (group === 'Color' && !job.color) return false;
    if (id === 'masks' && job.mask === 'off') return false;
    if ((id === 'global' || id === 'local') && job.exposure === 'off') return false;
    if (id === 'local' && job.exposure !== 'local') return false;
    return true;
  });
}
