#!/usr/bin/env python3
"""Canonical pipeline wrapper.

The analysis logic lives in src/01_audit_clean_features.py,
src/02_eda_charts.py and src/03_model_policy_analysis.py.
This file is only the orchestration entry point; it does not duplicate model logic.
"""
from pathlib import Path
import subprocess, sys, os

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

steps = [
    [sys.executable, str(SRC / "01_audit_clean_features.py")],
    [sys.executable, str(SRC / "02_eda_charts.py")],
    [sys.executable, str(SRC / "03_model_policy_analysis.py")],
    [sys.executable, str(SRC / "03b_shap_only.py")],
    [sys.executable, str(SRC / "04_robustness_and_fairness.py")],
]
for cmd in steps:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=SRC, check=True, env=os.environ.copy())
print("\nCanonical pipeline complete. See outputs/ and charts/.")
