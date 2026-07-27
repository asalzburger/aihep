"""Test setup for the soho scripts.

These are flat, cwd-relative scripts (not a package), so tests need a
little plumbing:

- `python/` is put on sys.path so `import svg_utils` works like it does
  when a script in `python/` imports its sibling module.
- `load_module()` loads a script by file path (needed for the numbered
  scripts, e.g. `01_extract_dots.py`, whose names aren't valid Python
  identifiers and so can't be `import`-ed normally).
- `project_copy` gives tests a throwaway copy of python/ + resources/ so
  scripts that write output files (SVGs, regenerated dots.csv) never touch
  the real working tree.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = PROJECT_ROOT / "python"
RESOURCES_DIR = PROJECT_ROOT / "resources"

sys.path.insert(0, str(PYTHON_DIR))


def load_module(path: Path, name: str) -> types.ModuleType:
    """Load a script as a module by file path, without running its
    `if __name__ == "__main__":` block (name won't be "__main__")."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def project_copy(tmp_path_factory) -> Path:
    """A throwaway copy of python/ + resources/, for subprocess smoke tests
    that run a whole script (and so write its output files) as documented."""
    dest = tmp_path_factory.mktemp("soho")
    shutil.copytree(PYTHON_DIR, dest / "python", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(RESOURCES_DIR, dest / "resources")
    return dest
