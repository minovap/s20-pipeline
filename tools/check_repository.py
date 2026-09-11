"""Check staged/tracked files before committing; never inspect private capture payloads."""

import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
failures = []
for name in filter(None, paths):
    path = root / name
    if path.is_symlink():
        failures.append(f"Symlink: {name}")
    if path.stat().st_size > 2_000_000:
        failures.append(f"Unexpected large file: {name}")
    if not name.startswith("apps/desktop/src-tauri/icons/") and path.suffix.lower() in (
        ".bag",
        ".las",
        ".laz",
        ".pcd",
        ".npy",
        ".bin",
        ".pth",
        ".pt",
        ".jpg",
        ".png",
        ".exe",
        ".dll",
    ):
        failures.append(f"Generated/private asset: {name}")
    if path.suffix in (".py", ".cpp", ".hpp", ".mm", ".metal"):
        text = path.read_text()
        if name != "tools/check_repository.py" and any(
            marker in text for marker in ("/Users/", "AGENTS.md", "work/s20-test/")
        ):
            failures.append(f"Workspace dependency: {name}")
if failures:
    raise SystemExit("\n".join(failures))
print(
    f"Checked {len(list(filter(None, paths)))} tracked files: no data assets, symlinks, large files or workspace-dependent code."
)
