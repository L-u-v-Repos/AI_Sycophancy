#!/usr/bin/env python3
"""Launch the interactive Web UI for Sycophancy Testbed."""
import os
import sys
from pathlib import Path

# If running outside the project virtual environment, auto-switch to .venv python
if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
    candidates = [
        Path(__file__).resolve().parent / ".venv" / "bin" / "python3",
        Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python3",
        Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python3",
    ]
    for venv_py in candidates:
        if venv_py.exists():
            os.execv(str(venv_py), [str(venv_py)] + sys.argv)

# Add repo to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sycophancy_testbed.web import start_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    start_server(port)
