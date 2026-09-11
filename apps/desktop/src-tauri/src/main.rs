#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
    path::PathBuf,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Mutex,
    },
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
    running: AtomicBool,
    pid: AtomicU32,
    previews: Mutex<HashMap<String, PathBuf>>,
}
fn root(state: &Engine) -> Result<PathBuf, String> {
    Ok(state.root.lock().map_err(|e| e.to_string())?.clone())
}
fn python(root: &PathBuf) -> Result<Command, String> {
    let p = root.join(".venv/bin/python");
    if !p.is_file() {
        return Err(
            "Pipeline environment missing. Select an installed s20-pipeline checkout in Settings."
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
#[tauri::command]
fn configuration(state: State<Engine>) -> Result<Value, String> {
    let p = root(&state)?;
    Ok(
        json!({"root":p,"ready":p.join(".venv/bin/python").is_file()&&p.join("build/s20_geometry").is_file(),"running":state.running.load(Ordering::SeqCst)}),
    )
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
        return Err("Select the installed s20-pipeline repository folder.".into());
    }
    let dest = app.path().app_config_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;
    std::fs::write(dest.join("engine.json"), serde_json::to_vec(&p).unwrap())
        .map_err(|e| e.to_string())?;
    *state.root.lock().map_err(|e| e.to_string())? = p;
    Ok(())
}
#[tauri::command]
async fn inspect_capture(path: String, state: State<'_, Engine>) -> Result<Value, String> {
    let p = root(&state)?;
    tauri::async_runtime::spawn_blocking(move || bridge(p, vec!["inspect".into(), path]))
        .await
        .map_err(|e| e.to_string())?
}
#[tauri::command]
async fn hardware(state: State<'_, Engine>) -> Result<Value, String> {
    let p = root(&state)?;
    tauri::async_runtime::spawn_blocking(move || bridge(p, vec!["hardware".into()]))
        .await
        .map_err(|e| e.to_string())?
}
#[derive(Deserialize, Serialize, Clone)]
struct JobOptions {
    capture: String,
    output: String,
    resources: String,
    memory_gb: f64,
    color: bool,
    mask: String,
    exposure: String,
    pose_refinement: bool,
    resume: bool,
}
fn job_args(o: &JobOptions) -> Result<Vec<String>, String> {
    if !["interactive", "balanced", "throughput"].contains(&o.resources.as_str())
        || !["person", "off"].contains(&o.mask.as_str())
        || !["local", "global", "off"].contains(&o.exposure.as_str())
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
#[tauri::command]
fn start_job(
    options: JobOptions,
    app: tauri::AppHandle,
    state: State<Engine>,
) -> Result<(), String> {
    let args = job_args(&options)?;
    let mut command = python(&root(&state)?)?;
    if state
        .running
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("Another processing job is already running".into());
    }
    let child = command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            state.running.store(false, Ordering::SeqCst);
            return Err(e.to_string());
        }
    };
    state.pid.store(child.id(), Ordering::SeqCst);
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let run_id = options.output;
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
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(mut value) = serde_json::from_str::<Value>(&line) {
                if value.is_object() {
                    value["run_id"] = json!(run_id);
                    let _ = app.emit("pipeline-event", value);
                }
            }
        }
        let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(-1);
        let state = app.state::<Engine>();
        state.pid.store(0, Ordering::SeqCst);
        state.running.store(false, Ordering::SeqCst);
        let _ = app.emit("job-exit", json!({"run_id":run_id,"code":code}));
    });
    Ok(())
}
#[tauri::command]
fn cancel_job(state: State<Engine>) -> Result<(), String> {
    let pid = state.pid.load(Ordering::SeqCst);
    if pid == 0 {
        return Ok(());
    }
    let result = unsafe { libc::kill(pid as i32, libc::SIGINT) };
    if result != 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    Ok(())
}
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
    if std::fs::metadata(p).map_err(|e| e.to_string())?.len() > 48_000_000 {
        return Err("Preview exceeds budget".into());
    }
    Ok(tauri::ipc::Response::new(
        std::fs::read(p).map_err(|e| e.to_string())?,
    ))
}
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let default = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(3)
                .unwrap()
                .to_path_buf();
            let settings = app.path().app_config_dir()?.join("engine.json");
            let root = std::fs::read(settings)
                .ok()
                .and_then(|b| serde_json::from_slice::<PathBuf>(&b).ok())
                .unwrap_or(default);
            app.manage(Engine {
                root: Mutex::new(root),
                running: AtomicBool::new(false),
                pid: AtomicU32::new(0),
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
                        "Cancel the active job before closing S20 Studio.",
                    );
                } else {
                    stop_auxiliary();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            configuration,
            configure,
            inspect_capture,
            hardware,
            start_job,
            cancel_job,
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
                        "Cancel the active job before quitting S20 Studio.",
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
            pose_refinement: true,
            resume: false,
        }
    }
    #[test]
    fn paths_are_literal_arguments() {
        let o = options();
        let args = job_args(&o).unwrap();
        assert_eq!(args[3], o.capture);
        assert_eq!(args[5], o.output);
        assert!(args.contains(&"--camera-convention".into()));
    }
    #[test]
    fn invalid_options_rejected() {
        let mut o = options();
        o.memory_gb = f64::NAN;
        assert!(job_args(&o).is_err());
        o.memory_gb = 16.;
        o.resources = "other".into();
        assert!(job_args(&o).is_err());
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
}
