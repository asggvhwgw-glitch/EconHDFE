import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_architecture_map_generated_outputs_are_current():
    subprocess.run(
        [sys.executable, str(ROOT / "skills" / "econhdfe" / "scripts" / "architecture_map.py"), "--check"],
        cwd=ROOT,
        check=True,
    )
