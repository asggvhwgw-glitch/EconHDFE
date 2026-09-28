"""The current package is the contract target; old snapshots are immutable history."""
import json
import subprocess
import sys
from pathlib import Path
import econhdfe
from scripts import compatibility

ROOT = Path(__file__).resolve().parents[2]


def test_current_snapshot_matches_live_public_contract():
    path = ROOT/'compatibility'/'baselines'/f'econhdfe-{econhdfe.__version__}.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    live = compatibility.capture_contract()
    assert live == frozen
    assert set(live['public_namespaces']) == {'econhdfe', 'econhdfe.effects'}
    assert live['error_catalog']['SpecificationError']['code'] == 'specification.invalid'


def test_current_release_review_gate_runs_as_a_public_command():
    # One real CLI invokes version, maintenance, compatibility, Skill and registry
    # checks. Negative unit tests exercise their independent failure contracts.
    result = subprocess.run([sys.executable, 'scripts/version.py', 'gate'],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == econhdfe.__version__
