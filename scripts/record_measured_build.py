"""Record reproducible inputs and outputs without claiming empirical validity."""
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/measured_cmj"


def main():
    paths = list((ROOT / "scripts").glob("*.py"))
    paths += list((ROOT / "scripts").glob("*.cjs"))
    paths += list((ROOT / "scenarios").glob("*.json"))
    paths += list((ROOT / "datasets/cmj_calisti2025").rglob("*"))
    paths += list(OUT.rglob("*"))
    paths += [ROOT / "open_measured_validation.cmd",
              ROOT / "rebuild_measured_validation.cmd", ROOT / "README.md"]
    files = {}
    for path in sorted(set(paths)):
        if not path.is_file() or path.name == "build_manifest.json":
            continue
        files[path.relative_to(ROOT).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    versions = {}
    for package in ("numpy", "scipy", "opensim", "mujoco", "ezc3d",
                    "matplotlib", "openpyxl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not exposed by package metadata"
    result = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "rebuild": "rebuild_measured_validation.cmd",
        "scope": "Analytical fixtures and independently scaled reference-model replay; human muscle-force validation remains incomplete.",
        "files": files,
    }
    (OUT / "build_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"recorded_files": len(files), "packages": versions}))


if __name__ == "__main__":
    main()
