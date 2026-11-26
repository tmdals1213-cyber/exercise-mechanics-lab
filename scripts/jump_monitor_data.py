"""Small crash-tolerant I/O shared by the local experiment supervisor and UI."""
from pathlib import Path
import json, os, time

ROOT = Path(__file__).resolve().parents[1]
MONITOR = ROOT / 'deliverables/jump_optimization_monitor'
RUN = ROOT / 'results/jump_overnight'

def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(.02 * (attempt + 1))
    os.replace(tmp, path)

def lines(path):
    try:
        out = []
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            try: out.append(json.loads(line))
            except ValueError: pass  # The writer may be midway through its last line.
        return out
    except OSError:
        return []

def event(kind, **data):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'events.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(dict(time=time.time(), kind=kind, **data), ensure_ascii=False) + '\n')
