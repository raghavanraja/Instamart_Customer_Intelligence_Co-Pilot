import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
steps=[
    [sys.executable,"generate_data.py"],
    [sys.executable,"src/audit_raw.py"],
    [sys.executable,"src/clean_data.py"],
    [sys.executable,"src/build_database.py"],
    [sys.executable,"src/part_a_analysis.py"],
    [sys.executable,"src/part_b_model.py"],
    [sys.executable,"src/part_c_triage.py"],
    [sys.executable,"src/part_d_rag_agent.py"],
]
for cmd in steps:
    print("\\n>>>"," ".join(cmd),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)
print("\\nALL PARTS COMPLETED")
