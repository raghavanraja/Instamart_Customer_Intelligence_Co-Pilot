from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
POLICIES = DATA / "policies"
OUTPUTS = ROOT / "outputs"
STATE = ROOT / "state"
DB = DATA / "instamart.db"

def ensure_dirs():
    for p in (RAW, CLEAN, POLICIES, OUTPUTS, STATE):
        p.mkdir(parents=True, exist_ok=True)

def write_jsonl(path, records, append=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
