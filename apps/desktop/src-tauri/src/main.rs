#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! S20 Studio shell. Owns settings, project folders, the pipeline process and
//! bounded point-cloud previews. All computation runs in Python/native workers;
//! this file never interpolates a shell command.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tauri::{Emitter, Manager, State};

static AUXILIARY_PIDS: Mutex<Vec<u32>> = Mutex::new(Vec::new());
fn stop_auxiliary() {
    if let Ok(pids) = AUXILIARY_PIDS.lock() {
        for pid in pids.iter() {
            unsafe {
                libc::kill(*pid as i32, libc::SIGTERM);
            }
        }
    }
}

struct Engine {
    root: Mutex<PathBuf>,
    projects: Mutex<PathBuf>,
    running: AtomicBool,
    pid: AtomicU32,
    export_pid: AtomicU32,
    exporting: AtomicBool,
    cancel_copy: AtomicBool,
    previews: Mutex<HashMap<String, PathBuf>>,
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.)
}
fn root(state: &Engine) -> Result<PathBuf, String> {
    Ok(state.root.lock().map_err(|e| e.to_string())?.clone())
}
fn projects_root(state: &Engine) -> Result<PathBuf, String> {
    Ok(state.projects.lock().map_err(|e| e.to_string())?.clone())
}
fn engine_ready(root: &Path) -> bool {
    root.join(".venv/bin/python").is_file() && root.join("build/s20_geometry").is_file()
}
fn python(root: &PathBuf) -> Result<Command, String> {
    let p = root.join(".venv/bin/python");
    if !p.is_file() {
        return Err(
            "Pipeline environment missing. Choose the installed s20-pipeline folder in Settings."
                .into(),
        );
    }
    let mut c = Command::new(p);
    c.current_dir(root).env("PYTHONUNBUFFERED", "1");
    Ok(c)
}
fn bridge(root: PathBuf, args: Vec<String>) -> Result<Value, String> {
    let child = python(&root)?
        .args(["-m", "s20_pipeline.desktop"])
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let pid = child.id();
    AUXILIARY_PIDS.lock().map_err(|e| e.to_string())?.push(pid);
    let output = child.wait_with_output();
    AUXILIARY_PIDS
        .lock()
        .map_err(|e| e.to_string())?
        .retain(|p| *p != pid);
    let output = output.map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(6000)
            .collect());
    }
    serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())
}
fn read_json(path: &Path) -> Option<Value> {
    fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
}
fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    fs::rename(temporary, path).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------- settings

#[derive(Serialize, Deserialize, Default)]
struct Settings {
    engine: Option<PathBuf>,
    projects: Option<PathBuf>,
}
fn settings_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.join("settings.json"))
}
fn save_settings(app: &tauri::AppHandle, state: &Engine) -> Result<(), String> {
    let value = Settings {
        engine: Some(root(state)?),
        projects: Some(projects_root(state)?),
    };
    fs::write(
        settings_path(app)?,
        serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

#[tauri::command]
fn settings(state: State<Engine>) -> Result<Value, String> {
    let engine = root(&state)?;
    Ok(json!({
        "engine_root": engine,
        "engine_ready": engine_ready(&engine),
        "projects_root": projects_root(&state)?,
        "running": state.running.load(Ordering::SeqCst),
    }))
}
#[tauri::command]
fn configure(path: String, app: tauri::AppHandle, state: State<Engine>) -> Result<(), String> {
    if state.running.load(Ordering::SeqCst) {
        return Err("Wait for or cancel the active job before changing the engine.".into());
    }
    let p = PathBuf::from(path)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if !p.join(".venv/bin/python").is_file() || !p.join("src/s20_pipeline/cli.py").is_file() {
        return Err("Choose the installed s20-pipeline repository folder.".into());
    }
    *state.root.lock().map_err(|e| e.to_string())? = p;
    save_settings(&app, &state)
}
#[tauri::command]
fn set_projects_root(
    path: String,
    app: tauri::AppHandle,
    state: State<Engine>,
) -> Result<(), String> {
    let p = PathBuf::from(path);
    fs::create_dir_all(&p).map_err(|e| e.to_string())?;
    *state.projects.lock().map_err(|e| e.to_string())? =
        p.canonicalize().map_err(|e| e.to_string())?;
    save_settings(&app, &state)
}
#[tauri::command]
async fn hardware(state: State<'_, Engine>) -> Result<Value, String> {
    let p = root(&state)?;
    tauri::async_runtime::spawn_blocking(move || bridge(p, vec!["hardware".into()]))
        .await
        .map_err(|e| e.to_string())?
}
#[tauri::command]
fn reveal(path: String) -> Result<(), String> {
    if !Path::new(&path).exists() {
        return Err("That folder no longer exists".into());
    }
    Command::new("open")
        .arg("-R")
        .arg(&path)
        .status()
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ---------------------------------------------------------------- projects

/// A project is a folder under the projects root holding `project.json`,
/// `runs/<run>/` pipeline output directories and `exports/` slice exports.
fn project_file(state: &Engine, project: &str) -> Result<(PathBuf, PathBuf), String> {
    let root = projects_root(state)?;
    let folder = PathBuf::from(project)
        .canonicalize()
        .map_err(|_| "Project folder not found".to_string())?;
    if !folder.starts_with(&root) || folder == root {
        return Err("Project is outside the projects folder".into());
    }
    let file = folder.join("project.json");
    if !file.is_file() {
        return Err("Not a project folder".into());
    }
    Ok((folder, file))
}
fn slug(name: &str) -> String {
    let s: String = name
        .trim()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' { c } else { '-' })
        .collect();
    let s = s.trim().to_string();
    if s.is_empty() {
        "Project".into()
    } else {
        s
    }
}
fn summarize(folder: &Path) -> Option<Value> {
    let mut project = read_json(&folder.join("project.json"))?;
    let runs = read_runs(folder);
    let last = runs.first().cloned();
    project["path"] = json!(folder);
    project["run_count"] = json!(runs.len());
    project["last_run"] = last.unwrap_or(Value::Null);
    project["input_count"] = json!(project["inputs"].as_array().map(|a| a.len()).unwrap_or(0));
    Some(project)
}

/// Reconstruct a run's state from the pipeline's own on-disk records.
fn read_run(folder: &Path) -> Option<Value> {
    let job = read_json(&folder.join("job.json"))?;
    let state = read_json(&folder.join("state.json")).unwrap_or(json!({}));
    let mut stages: Vec<Value> = Vec::new();
    let mut started: Option<f64> = None;
    let mut finished: Option<f64> = None;
    let mut order: Vec<String> = Vec::new();
    if let Ok(text) = fs::read_to_string(folder.join("events.jsonl")) {
        for line in text.lines() {
            let Ok(event) = serde_json::from_str::<Value>(line) else { continue };
            let time = event["time_unix"].as_f64();
            if started.is_none() {
                started = time;
            }
            let kind = event["event"].as_str().unwrap_or("");
            if kind == "stage_started" || kind == "stage_cached" {
                if let Some(stage) = event["stage"].as_str() {
                    order.push(stage.to_string());
                }
            }
            if matches!(kind, "completed" | "failed" | "cancelled") {
                finished = time;
            }
        }
    }
    for stage in &order {
        let receipt = read_json(&folder.join("receipts").join(format!("{stage}.json")));
        stages.push(json!({
            "id": stage,
            "status": if receipt.is_some() { "complete" } else { "incomplete" },
            "wall_s": receipt.as_ref().and_then(|r| r["wall_s"].as_f64()),
        }));
    }
    let status = state["status"].as_str().unwrap_or("unknown");
    let color = job["options"]["color"].as_bool().unwrap_or(true);
    let result = if color {
        folder.join("export/colorized.las")
    } else {
        folder.join("geometry/filtered.ply")
    };
    let result_exists = result.is_file() && status == "completed";
    let copy = read_json(&folder.join("copy.json")).or_else(|| read_json(&folder.with_extension("copy.json")));
    let capture = copy.as_ref().and_then(|c| c["source"].as_str().map(|s| json!(s))).unwrap_or(job["capture"].clone());
    Some(json!({
        "path": folder,
        "name": folder.file_name().and_then(|n| n.to_str()).unwrap_or(""),
        "status": if status == "running" { "interrupted" } else { status },
        "stage": state["stage"],
        "error": state["error"],
        "started": started,
        "finished": finished,
        "capture": capture,
        "options": {
            "copy": copy.is_some(),
            "color": color,
            "mask": job["options"]["mask"],
            "exposure": job["options"]["exposure"],
            "pose_refinement": job["options"]["pose_refinement"],
            "resources": job["options"]["resources"],
            "memory_gb": job["options"]["memory_gb"],
            "photo_matching": job["options"]["photo_matching"].as_str().unwrap_or("exact"),
            "keyframe_percent": job["options"]["keyframe_percent"].as_u64().unwrap_or(30),
        },
        "stages": stages,
        "result": if result_exists { json!(result) } else { Value::Null },
        "result_points": read_json(&folder.join("export/result.json")).and_then(|r| r["colored_points"].as_u64()),
    }))
}
fn read_runs(folder: &Path) -> Vec<Value> {
    let mut runs: Vec<Value> = fs::read_dir(folder.join("runs"))
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .filter_map(|e| read_run(&e.path()))
                .collect()
        })
        .unwrap_or_default();
    runs.sort_by(|a, b| b["name"].as_str().cmp(&a["name"].as_str()));
    runs
}
fn read_exports(folder: &Path) -> Vec<Value> {
    let mut files: Vec<Value> = fs::read_dir(folder.join("exports"))
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let p = e.path();
                    p.is_file()
                        && matches!(
                            p.extension().and_then(|x| x.to_str()).map(|x| x.to_ascii_lowercase()),
                            Some(ref x) if x == "las" || x == "ply"
                        )
                })
                .map(|e| {
                    let p = e.path();
                    let meta = e.metadata().ok();
                    json!({
                        "path": p,
                        "name": p.file_stem().and_then(|n| n.to_str()).unwrap_or(""),
                        "bytes": meta.as_ref().map(|m| m.len()),
                        "modified": meta.and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs_f64()),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort_by(|a, b| b["modified"].as_f64().partial_cmp(&a["modified"].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
    files
}
fn full_project(folder: &Path) -> Result<Value, String> {
    let mut project = read_json(&folder.join("project.json")).ok_or("Project file unreadable")?;
    project["path"] = json!(folder);
    project["runs"] = json!(read_runs(folder));
    project["exports"] = json!(read_exports(folder));
    if !project["inputs"].is_array() {
        project["inputs"] = json!([]);
    }
    if !project["slices"].is_array() {
        project["slices"] = json!([]);
    }
    if !project["clouds"].is_array() {
        project["clouds"] = json!([]);
    }
    if !project["orientations"].is_object() {
        project["orientations"] = json!({});
    }
    Ok(project)
}

#[tauri::command]
fn list_projects(state: State<Engine>) -> Result<Vec<Value>, String> {
    let root = projects_root(&state)?;
    let mut projects: Vec<Value> = fs::read_dir(&root)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .filter_map(|e| summarize(&e.path()))
                .collect()
        })
        .unwrap_or_default();
    projects.sort_by(|a, b| {
        let ta = a["last_run"]["started"].as_f64().or(a["created"].as_f64()).unwrap_or(0.);
        let tb = b["last_run"]["started"].as_f64().or(b["created"].as_f64()).unwrap_or(0.);
        tb.partial_cmp(&ta).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(projects)
}
#[tauri::command]
fn create_project(name: String, state: State<Engine>) -> Result<Value, String> {
    let root = projects_root(&state)?;
    fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let base = slug(&name);
    let mut folder = root.join(&base);
    let mut n = 2;
    while folder.exists() {
        folder = root.join(format!("{base} {n}"));
        n += 1;
    }
    fs::create_dir_all(folder.join("runs")).map_err(|e| e.to_string())?;
    fs::create_dir_all(folder.join("exports")).map_err(|e| e.to_string())?;
    write_json(
        &folder.join("project.json"),
        &json!({
            "schema": 1,
            "name": if name.trim().is_empty() { base.clone() } else { name.trim().to_string() },
            "created": now(),
            "inputs": [],
            "clouds": [],
            "slices": [],
        }),
    )?;
    full_project(&folder)
}
#[tauri::command]
fn open_project(path: String, state: State<Engine>) -> Result<Value, String> {
    let (folder, _) = project_file(&state, &path)?;
    full_project(&folder)
}
/// Persist the editable part of a project: name, inputs, clouds, slices.
#[tauri::command]
fn write_project(path: String, project: Value, state: State<Engine>) -> Result<Value, String> {
    let (folder, file) = project_file(&state, &path)?;
    let mut stored = read_json(&file).unwrap_or(json!({"schema":1}));
    for key in ["name", "inputs", "clouds", "slices", "orientations"] {
        if !project[key].is_null() {
            stored[key] = project[key].clone();
        }
    }
    write_json(&file, &stored)?;
    full_project(&folder)
}
#[tauri::command]
async fn add_input(
    project: String,
    capture: String,
    state: State<'_, Engine>,
) -> Result<Value, String> {
    let (folder, file) = project_file(&state, &project)?;
    let p = root(&state)?;
    let inspected = tauri::async_runtime::spawn_blocking(move || {
        bridge(p, vec!["inspect".into(), capture])
    })
    .await
    .map_err(|e| e.to_string())??;
    let capture_path = inspected["capture"]["capture"]
        .as_str()
        .ok_or("Inspection returned no capture path")?
        .to_string();
    let mut stored = read_json(&file).ok_or("Project file unreadable")?;
    let mut inputs = stored["inputs"].as_array().cloned().unwrap_or_default();
    inputs.retain(|i| i["path"].as_str() != Some(capture_path.as_str()));
    inputs.push(json!({
        "path": capture_path,
        "name": Path::new(&capture_path).file_name().and_then(|n| n.to_str()).unwrap_or("capture"),
        "added": now(),
        "capture": inspected["capture"],
        "estimate": inspected["estimate"],
    }));
    stored["inputs"] = json!(inputs);
    write_json(&file, &stored)?;
    full_project(&folder)
}
#[tauri::command]
fn input_available(path: String) -> bool {
    Path::new(&path).is_dir()
}
#[tauri::command]
fn read_stage_log(run: String, stage: String, state: State<Engine>) -> Result<String, String> {
    let root = projects_root(&state)?;
    let folder = PathBuf::from(&run).canonicalize().map_err(|e| e.to_string())?;
    if !folder.starts_with(&root) {
        return Err("Run is outside the projects folder".into());
    }
    if stage.contains('/') || stage.contains("..") {
        return Err("Invalid stage".into());
    }
    let path = folder.join("logs").join(format!("{stage}.log"));
    let bytes = fs::read(&path).map_err(|_| "No log for this step yet".to_string())?;
    let tail = if bytes.len() > 65536 { &bytes[bytes.len() - 65536..] } else { &bytes[..] };
    Ok(String::from_utf8_lossy(tail).into_owned())
}
#[tauri::command]
fn delete_run(run: String, state: State<Engine>) -> Result<(), String> {
    if state.running.load(Ordering::SeqCst) {
        return Err("Wait for the active job to finish first".into());
    }
    let root = projects_root(&state)?;
    let folder = PathBuf::from(&run).canonicalize().map_err(|e| e.to_string())?;
    let is_run = folder.starts_with(&root)
        && folder.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()) == Some("runs")
        && folder.join("job.json").is_file();
    if !is_run {
        return Err("Not a run folder".into());
    }
    if let Some(temp) = read_json(&folder.join("copy.json")).and_then(|c| c["temp"].as_str().map(|t| t.to_string())) {
        let shared = folder.parent().and_then(|runs| fs::read_dir(runs).ok()).map(|entries| {
            entries.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p != &folder)
                .filter_map(|p| read_json(&p.join("copy.json")))
                .any(|c| c["temp"].as_str() == Some(temp.as_str()))
        }).unwrap_or(false);
        if !shared { remove_temp_copy(Path::new(&temp)); }
    }
    fs::remove_dir_all(folder).map_err(|e| e.to_string())
}
#[tauri::command]
fn delete_export(path: String, state: State<Engine>) -> Result<(), String> {
    let root = projects_root(&state)?;
    let file = PathBuf::from(&path).canonicalize().map_err(|e| e.to_string())?;
    let ok = file.starts_with(&root)
        && file.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()) == Some("exports")
        && file.is_file();
    if !ok {
        return Err("Not an export file".into());
    }
    fs::remove_file(file).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------- jobs

fn default_photo_matching() -> String { "exact".into() }
fn default_keyframe_percent() -> u32 { 30 }

#[derive(Deserialize, Serialize, Clone)]
struct JobOptions {
    capture: String,
    output: String,
    resources: String,
    memory_gb: f64,
    color: bool,
    mask: String,
    exposure: String,
    #[serde(default = "default_photo_matching")]
    photo_matching: String,
    #[serde(default = "default_keyframe_percent")]
    keyframe_percent: u32,
    pose_refinement: bool,
    resume: bool,
    #[serde(default)]
    copy: bool,
}
fn job_args(o: &JobOptions) -> Result<Vec<String>, String> {
    if !["interactive", "balanced", "throughput"].contains(&o.resources.as_str())
        || !["person", "off"].contains(&o.mask.as_str())
        || !["local", "global", "off"].contains(&o.exposure.as_str())
        || !["exact", "keyframes"].contains(&o.photo_matching.as_str())
        || !(1..=100).contains(&o.keyframe_percent)
        || !o.memory_gb.is_finite()
        || o.memory_gb < 1.
        || o.memory_gb > 1024.
    {
        return Err("Invalid processing options".into());
    }
    let mut a = vec![
        "-m".into(),
        "s20_pipeline.cli".into(),
        "run".into(),
        o.capture.clone(),
        "--output".into(),
        o.output.clone(),
        "--resources".into(),
        o.resources.clone(),
        "--memory-gb".into(),
        o.memory_gb.to_string(),
        "--mask".into(),
        o.mask.clone(),
        "--exposure".into(),
        o.exposure.clone(),
        "--photo-matching".into(),
        o.photo_matching.clone(),
        "--keyframe-percent".into(),
        o.keyframe_percent.to_string(),
    ];
    if o.color {
        a.extend([
            "--camera-clock".into(),
            "sensor-header".into(),
            "--camera-convention".into(),
            "lidar-extrinsics".into(),
        ]);
    } else {
        a.push("--no-color".into());
    }
    if !o.pose_refinement {
        a.push("--no-pose-refinement".into());
    }
    if o.resume {
        a.push("--resume".into());
    }
    Ok(a)
}
/// Identity of a scan folder from its file names, sizes and modification
/// times (FNV-1a, 12 hex characters). The same scan always maps to the same
/// temporary copy, so later runs and resumes find it without copying again.
fn scan_identity(source: &Path) -> Result<String, String> {
    let mut files = Vec::new();
    walk(source, &mut files)?;
    let mut rows: Vec<(String, u64, u128)> = files
        .iter()
        .map(|(path, size)| {
            let relative = path.strip_prefix(source).map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
            let mtime = fs::metadata(path).ok().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_nanos()).unwrap_or(0);
            (relative, *size, mtime)
        })
        .collect();
    rows.sort();
    let mut hash: u64 = 0xcbf29ce484222325;
    for (name, size, mtime) in rows {
        for byte in format!("{name}\0{size}\0{mtime}\n").bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{hash:016x}")[..12].to_string())
}
/// Where a scan's temporary copy lives: ~/Downloads/S20 temp/<scan>-<identity>.
fn temp_copy_dir(app: &tauri::AppHandle, capture: &str) -> Result<PathBuf, String> {
    let source = Path::new(capture);
    let scan = source.file_name().and_then(|n| n.to_str()).ok_or("Invalid scan folder")?;
    let downloads = app.path().download_dir().map_err(|e| e.to_string())?;
    Ok(downloads.join("S20 temp").join(format!("{scan}-{}", scan_identity(source)?)))
}
/// Files of a scan folder, skipping hidden entries such as Finder's .DS_Store.
fn walk(dir: &Path, out: &mut Vec<(PathBuf, u64)>) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if entry.file_name().to_string_lossy().starts_with('.') { continue; }
        let meta = entry.metadata().map_err(|e| e.to_string())?;
        if meta.is_dir() { walk(&path, out)?; } else if meta.is_file() { out.push((path, meta.len())); }
    }
    Ok(())
}
/// Copy a scan folder with progress events. Modification times are kept so
/// the pipeline's size-and-mtime input identity matches on resume. Files
/// already present with the same size and mtime are skipped.
fn copy_scan(app: &tauri::AppHandle, state: &Engine, source: &Path, dest: &Path, run_id: &str) -> Result<(), String> {
    let mut files = Vec::new();
    walk(source, &mut files)?;
    let total: u64 = files.iter().map(|(_, n)| n).sum();
    let mut done: u64 = 0;
    let mut last = std::time::Instant::now();
    let emit = |event: &str, extra: Value| {
        let mut v = json!({"event": event, "stage": "copy", "run_id": run_id, "time_unix": now()});
        if let Some(o) = extra.as_object() { for (k, val) in o { v[k] = val.clone(); } }
        let _ = app.emit("pipeline-event", v);
    };
    emit("stage_started", json!({}));
    emit("progress", json!({"done": 0, "total": total, "unit": "bytes"}));
    let started = std::time::Instant::now();
    let mut buffer = vec![0u8; 8 << 20];
    // Progress counts bytes flushed to disk, not bytes read: a source that is
    // still in the file cache reads at memory speed and would race ahead.
    const SYNC_EVERY: u64 = 64 << 20;
    for (path, size) in &files {
        if state.cancel_copy.load(Ordering::SeqCst) { return Err("cancelled".into()); }
        let relative = path.strip_prefix(source).map_err(|e| e.to_string())?;
        let target = dest.join(relative);
        if let Some(parent) = target.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
        let src_meta = fs::metadata(path).map_err(|e| e.to_string())?;
        let modified = src_meta.modified().map_err(|e| e.to_string())?;
        if let Ok(existing) = fs::metadata(&target) {
            if existing.len() == *size && existing.modified().ok() == Some(modified) { done += size; continue; }
        }
        let mut reader = fs::File::open(path).map_err(|e| e.to_string())?;
        let mut writer = fs::File::create(&target).map_err(|e| e.to_string())?;
        use std::io::{Read, Write};
        let file_start = done;
        let mut written: u64 = 0;
        let mut unsynced: u64 = 0;
        loop {
            if state.cancel_copy.load(Ordering::SeqCst) { drop(writer); let _ = fs::remove_file(&target); return Err("cancelled".into()); }
            let n = reader.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 { break; }
            writer.write_all(&buffer[..n]).map_err(|e| e.to_string())?;
            written += n as u64;
            unsynced += n as u64;
            if unsynced >= SYNC_EVERY {
                writer.sync_data().map_err(|e| e.to_string())?;
                unsynced = 0;
                done = file_start + written;
                if last.elapsed().as_millis() > 400 {
                    last = std::time::Instant::now();
                    emit("progress", json!({"done": done, "total": total, "unit": "bytes"}));
                }
            }
        }
        writer.sync_all().map_err(|e| e.to_string())?;
        writer.set_modified(modified).map_err(|e| e.to_string())?;
        done = file_start + written;
    }
    emit("progress", json!({"done": total, "total": total, "unit": "bytes"}));
    emit("stage_completed", json!({"wall_s": started.elapsed().as_secs_f64()}));
    Ok(())
}
fn remove_temp_copy(temp: &Path) {
    let _ = fs::remove_dir_all(temp);
}

#[tauri::command]
fn start_job(
    options: JobOptions,
    app: tauri::AppHandle,
    state: State<Engine>,
) -> Result<(), String> {
    let mut effective = options.clone();
    let temp = if options.copy {
        let dir = temp_copy_dir(&app, &options.capture)?;
        effective.capture = dir.to_string_lossy().into_owned();
        Some(dir)
    } else { None };
    let args = job_args(&effective)?;
    let mut command = python(&root(&state)?)?;
    if state
        .running
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("Another processing job is already running".into());
    }
    if let Some(parent) = Path::new(&options.output).parent() {
        let _ = fs::create_dir_all(parent);
    }
    state.cancel_copy.store(false, Ordering::SeqCst);
    let run_id = options.output.clone();
    let source = PathBuf::from(options.capture.clone());
    let output = options.output.clone();
    std::thread::spawn(move || {
        let state = app.state::<Engine>();
        if let Some(dir) = &temp {
            // The run folder is created by the pipeline itself, so the copy
            // record lives next to it until the run folder exists.
            let _ = write_json(&Path::new(&output).with_extension("copy.json"), &json!({"source": source, "temp": effective.capture}));
            if let Err(e) = copy_scan(&app, &state, &source, dir, &run_id) {
                let cancelled = e == "cancelled";
                let _ = app.emit("pipeline-event", json!({"event": if cancelled { "cancelled" } else { "failed" }, "stage": "copy", "run_id": run_id, "message": if cancelled { "Copy cancelled".to_string() } else { format!("Copying the scan failed: {e}") }, "time_unix": now()}));
                state.running.store(false, Ordering::SeqCst);
                let _ = app.emit("job-exit", json!({"run_id": run_id, "code": if cancelled { 130 } else { 1 }}));
                return;
            }
        }
        let child = command
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                let _ = app.emit("job-log", json!({"run_id": run_id, "message": e.to_string()}));
                state.running.store(false, Ordering::SeqCst);
                let _ = app.emit("job-exit", json!({"run_id": run_id, "code": 1}));
                return;
            }
        };
        state.pid.store(child.id(), Ordering::SeqCst);
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let errapp = app.clone();
        let errid = run_id.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let _ = errapp.emit(
                    "job-log",
                    json!({"run_id":errid,"message":line.chars().take(3000).collect::<String>()}),
                );
            }
        });
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(mut value) = serde_json::from_str::<Value>(&line) {
                if value.is_object() {
                    value["run_id"] = json!(run_id);
                    let _ = app.emit("pipeline-event", value);
                }
            }
        }
        let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(-1);
        state.pid.store(0, Ordering::SeqCst);
        state.running.store(false, Ordering::SeqCst);
        if temp.is_some() {
            // Move the copy record into the run folder now that it exists.
            let sidecar = Path::new(&output).with_extension("copy.json");
            let _ = write_json(&Path::new(&output).join("copy.json"), &json!({"source": source, "temp": effective.capture}));
            let _ = fs::remove_file(sidecar);
            if code == 0 { remove_temp_copy(Path::new(&effective.capture)); }
        }
        let _ = app.emit("job-exit", json!({"run_id":run_id,"code":code}));
    });
    Ok(())
}
#[tauri::command]
fn cancel_job(state: State<Engine>) -> Result<(), String> {
    let pid = state.pid.load(Ordering::SeqCst);
    if pid == 0 {
        state.cancel_copy.store(true, Ordering::SeqCst);
        return Ok(());
    }
    let result = unsafe { libc::kill(pid as i32, libc::SIGINT) };
    if result != 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    Ok(())
}

// ---------------------------------------------------------------- slice export

/// Run the Python slice exporter in the background. Progress arrives as
/// `export-event` with the export id; the spec is written to a temporary file
/// so no coordinates pass through argv.
#[tauri::command]
fn export_slices(
    spec: Value,
    app: tauri::AppHandle,
    state: State<Engine>,
) -> Result<String, String> {
    let output = spec["output"].as_str().ok_or("Export needs an output path")?;
    let root = projects_root(&state)?;
    let out = PathBuf::from(output);
    let parent = out.parent().ok_or("Invalid output path")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let parent = parent.canonicalize().map_err(|e| e.to_string())?;
    if !parent.starts_with(&root) && !spec["allow_outside"].as_bool().unwrap_or(false) {
        return Err("Export location must be inside a project".into());
    }
    if state
        .exporting
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("Another export is still running".into());
    }
    let id = format!("export-{}", now());
    let cache = app.path().app_cache_dir().map_err(|e| e.to_string())?;
    fs::create_dir_all(&cache).map_err(|e| e.to_string())?;
    let spec_path = cache.join(format!("{id}.json"));
    if let Err(e) = fs::write(&spec_path, serde_json::to_vec(&spec).unwrap()) {
        state.exporting.store(false, Ordering::SeqCst);
        return Err(e.to_string());
    }
    let mut command = match python(&root_or_reset(&state)) {
        Ok(c) => c,
        Err(e) => {
            state.exporting.store(false, Ordering::SeqCst);
            return Err(e);
        }
    };
    let child = command
        .args(["-m", "s20_pipeline.desktop", "export-slices"])
        .arg(&spec_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            state.exporting.store(false, Ordering::SeqCst);
            return Err(e.to_string());
        }
    };
    state.export_pid.store(child.id(), Ordering::SeqCst);
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let export_id = id.clone();
    let name = out.file_stem().and_then(|n| n.to_str()).unwrap_or("export").to_string();
    std::thread::spawn(move || {
        let mut last_error = String::new();
        let err_thread = std::thread::spawn(move || {
            let mut text = String::new();
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                text = line;
            }
            text
        });
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(mut value) = serde_json::from_str::<Value>(&line) {
                if value.is_object() {
                    value["id"] = json!(export_id);
                    value["name"] = json!(name);
                    let _ = app.emit("export-event", value);
                }
            }
        }
        let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(-1);
        if let Ok(text) = err_thread.join() {
            last_error = text;
        }
        let state = app.state::<Engine>();
        state.export_pid.store(0, Ordering::SeqCst);
        state.exporting.store(false, Ordering::SeqCst);
        let _ = fs::remove_file(&spec_path);
        if code != 0 {
            let message = if last_error.is_empty() {
                format!("Export exited with code {code}")
            } else {
                last_error.trim_start_matches("s20: ").to_string()
            };
            let _ = app.emit(
                "export-event",
                json!({"id": export_id, "name": name, "event": if code == 130 || code == -1 { "cancelled" } else { "failed" }, "message": message}),
            );
        }
    });
    Ok(id)
}
fn root_or_reset(state: &Engine) -> PathBuf {
    root(state).unwrap_or_default()
}
#[tauri::command]
fn cancel_export(state: State<Engine>) -> Result<(), String> {
    let pid = state.export_pid.load(Ordering::SeqCst);
    if pid == 0 {
        return Ok(());
    }
    unsafe {
        libc::kill(pid as i32, libc::SIGTERM);
    }
    Ok(())
}

// ---------------------------------------------------------------- previews

#[tauri::command]
async fn load_preview(
    source: String,
    budget: u32,
    app: tauri::AppHandle,
    state: State<'_, Engine>,
) -> Result<Value, String> {
    let p = root(&state)?;
    let cache = app
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?
        .join("previews");
    let value = tauri::async_runtime::spawn_blocking(move || {
        bridge(
            p,
            vec![
                "preview".into(),
                source,
                cache.to_string_lossy().into(),
                "--budget".into(),
                budget.to_string(),
            ],
        )
    })
    .await
    .map_err(|e| e.to_string())??;
    let path = PathBuf::from(value["file"].as_str().ok_or("No preview file")?);
    let allowed = app
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    if !path.starts_with(allowed) {
        return Err("Preview outside cache".into());
    }
    let mut paths = state.previews.lock().map_err(|e| e.to_string())?;
    if paths.len() > 8 {
        paths.clear();
    }
    paths.insert(value["key"].as_str().ok_or("No preview key")?.into(), path);
    Ok(value)
}
#[tauri::command]
fn read_preview(key: String, state: State<Engine>) -> Result<tauri::ipc::Response, String> {
    let paths = state.previews.lock().map_err(|e| e.to_string())?;
    let p = paths.get(&key).ok_or("Preview expired; load again")?;
    if fs::metadata(p).map_err(|e| e.to_string())?.len() > 8_000_000 * 24 {
        return Err("Preview exceeds budget".into());
    }
    Ok(tauri::ipc::Response::new(
        fs::read(p).map_err(|e| e.to_string())?,
    ))
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let default_engine = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(3)
                .unwrap()
                .to_path_buf();
            let config = app.path().app_config_dir()?;
            let stored: Settings = fs::read(config.join("settings.json"))
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok())
                .unwrap_or_default();
            // Earlier builds stored only the engine path.
            let legacy_engine = fs::read(config.join("engine.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<PathBuf>(&b).ok());
            let projects = stored.projects.unwrap_or_else(|| {
                app.path()
                    .document_dir()
                    .map(|d| d.join("S20 Projects"))
                    .unwrap_or_else(|_| PathBuf::from("S20 Projects"))
            });
            let _ = fs::create_dir_all(&projects);
            app.manage(Engine {
                root: Mutex::new(stored.engine.or(legacy_engine).unwrap_or(default_engine)),
                projects: Mutex::new(projects),
                running: AtomicBool::new(false),
                pid: AtomicU32::new(0),
                export_pid: AtomicU32::new(0),
                exporting: AtomicBool::new(false),
                cancel_copy: AtomicBool::new(false),
                previews: Mutex::new(HashMap::new()),
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if window.state::<Engine>().running.load(Ordering::SeqCst) {
                    api.prevent_close();
                    let _ = window.emit(
                        "close-blocked",
                        "Processing is still running. Cancel it before closing.",
                    );
                } else {
                    stop_auxiliary();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            settings,
            configure,
            set_projects_root,
            hardware,
            reveal,
            list_projects,
            create_project,
            open_project,
            write_project,
            add_input,
            input_available,
            read_stage_log,
            delete_run,
            delete_export,
            start_job,
            cancel_job,
            export_slices,
            cancel_export,
            load_preview,
            read_preview
        ])
        .build(tauri::generate_context!())
        .expect("Unable to start S20 Studio")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { api, .. } = event {
                if app.state::<Engine>().running.load(Ordering::SeqCst) {
                    api.prevent_exit();
                    let _ = app.emit(
                        "close-blocked",
                        "Processing is still running. Cancel it before quitting.",
                    );
                } else {
                    stop_auxiliary();
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options() -> JobOptions {
        JobOptions {
            capture: "/path with spaces/$(literal)".into(),
            output: "/output folder/run".into(),
            resources: "throughput".into(),
            memory_gb: 16.,
            color: true,
            mask: "person".into(),
            exposure: "local".into(),
            photo_matching: "exact".into(),
            keyframe_percent: 30,
            pose_refinement: true,
            resume: false,
            copy: false,
        }
    }
    #[test]
    fn paths_are_literal_arguments() {
        let o = options();
        let args = job_args(&o).unwrap();
        assert_eq!(args[3], o.capture);
        assert_eq!(args[5], o.output);
        assert!(args.contains(&"--camera-convention".into()));
        assert!(args.windows(2).any(|a| a == ["--photo-matching", "exact"]));
        assert!(args.windows(2).any(|a| a == ["--keyframe-percent", "30"]));
    }
    #[test]
    fn invalid_options_rejected() {
        let mut o = options();
        o.memory_gb = f64::NAN;
        assert!(job_args(&o).is_err());
        o.memory_gb = 16.;
        o.resources = "other".into();
        assert!(job_args(&o).is_err());
        o.resources = "balanced".into();
        o.photo_matching = "other".into();
        assert!(job_args(&o).is_err());
        o.photo_matching = "keyframes".into();
        o.keyframe_percent = 0;
        assert!(job_args(&o).is_err());
    }
    #[test]
    fn fast_photo_matching_is_passed_to_pipeline() {
        let mut o = options();
        o.photo_matching = "keyframes".into();
        let args = job_args(&o).unwrap();
        assert!(args.windows(2).any(|a| a == ["--photo-matching", "keyframes"]));
        let old = serde_json::json!({
            "capture": "/scan", "output": "/run", "resources": "balanced",
            "memory_gb": 16.0, "color": true, "mask": "person", "exposure": "local",
            "pose_refinement": true, "resume": false
        });
        let parsed: JobOptions = serde_json::from_value(old).unwrap();
        assert_eq!(parsed.photo_matching, "exact");
        assert_eq!(parsed.keyframe_percent, 30);
    }
    #[test]
    fn geometry_only_and_resume_are_explicit() {
        let mut o = options();
        o.color = false;
        o.resume = true;
        let a = job_args(&o).unwrap();
        assert!(a.contains(&"--no-color".into()));
        assert!(a.contains(&"--resume".into()));
        assert!(!a.contains(&"--camera-clock".into()));
    }
    #[test]
    fn slugs_keep_names_readable() {
        assert_eq!(slug("  Garden / north  "), "Garden - north");
        assert_eq!(slug(""), "Project");
    }
    #[test]
    fn scan_identity_depends_on_names_sizes_and_times() {
        let dir = std::env::temp_dir().join(format!("s20-scan-{}", now()));
        fs::create_dir_all(dir.join("info")).unwrap();
        fs::write(dir.join("all.bag"), b"abc").unwrap();
        fs::write(dir.join("info/calibration.yaml"), b"x").unwrap();
        let stamp = UNIX_EPOCH + std::time::Duration::from_nanos(1_700_000_000_000_000_000);
        for f in ["all.bag", "info/calibration.yaml"] { fs::File::options().write(true).open(dir.join(f)).unwrap().set_modified(stamp).unwrap(); }
        fs::write(dir.join(".DS_Store"), b"ignored").unwrap();
        let a = scan_identity(&dir).unwrap();
        assert_eq!(a, "73b9e7488f43");
        assert_eq!(a, scan_identity(&dir).unwrap());
        fs::write(dir.join("all.bag"), b"abcd").unwrap();
        assert_ne!(a, scan_identity(&dir).unwrap());
        let _ = fs::remove_dir_all(dir);
    }
    #[test]
    fn runs_are_reconstructed_from_disk() {
        let dir = std::env::temp_dir().join(format!("s20-run-{}", now()));
        fs::create_dir_all(dir.join("receipts")).unwrap();
        fs::write(dir.join("job.json"), r#"{"capture":"/c","options":{"color":false,"mask":"person","exposure":"local","pose_refinement":true,"resources":"balanced","memory_gb":16}}"#).unwrap();
        fs::write(dir.join("state.json"), r#"{"status":"cancelled","stage":"pack","error":"stopped"}"#).unwrap();
        fs::write(dir.join("events.jsonl"), "{\"event\":\"stage_started\",\"time_unix\":10.0,\"stage\":\"decode\"}\n{\"event\":\"stage_completed\",\"time_unix\":15.0,\"stage\":\"decode\",\"wall_s\":5.0}\n{\"event\":\"stage_started\",\"time_unix\":15.0,\"stage\":\"pack\"}\n{\"event\":\"cancelled\",\"time_unix\":17.0,\"stage\":\"pack\"}\n").unwrap();
        fs::write(dir.join("receipts/decode.json"), r#"{"wall_s":5.0}"#).unwrap();
        let run = read_run(&dir).unwrap();
        assert_eq!(run["status"], "cancelled");
        assert_eq!(run["started"], 10.0);
        assert_eq!(run["finished"], 17.0);
        assert_eq!(run["stages"][0]["status"], "complete");
        assert_eq!(run["stages"][0]["wall_s"], 5.0);
        assert_eq!(run["stages"][1]["status"], "incomplete");
        assert!(run["result"].is_null());
        let _ = fs::remove_dir_all(dir);
    }
}
