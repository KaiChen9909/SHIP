"""Read-only dataset file sandbox for evaluation code agents.

Scope: this guards against an evaluation agent *accidentally* reading files other than the
released insight (e.g. `private_DO_NOT_RELEASE/`), writing files, or leaving the dataset
directory. Agent code runs in-process via `exec`, so this is a best-effort guard for a
cooperative model, not a security boundary against adversarial code.
"""

from __future__ import annotations

import builtins
import contextvars
import io
import json
import os
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DATA_DIR: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_data_dir", default=None)

ALLOWED_DATASET_FILENAMES: frozenset[str] = frozenset({
    "synthetic.csv",
    "taxonomy.json",
    "category_tree.json",
    "synthetic_allocated_paths_divided.csv",
    "synthetic_allocated_paths_divided.json",
    "synthetic_allocated_paths.csv",
    "synthetic_allocated_paths.json",
    "allocated_paths_divided.csv",
    "allocated_paths_divided.json",
    "allocated_paths.csv",
    "allocated_paths.json",
    "dataset_info.json",
})


def resolve_sandboxed_path(path_or_name: Any) -> str:
    """Resolve a path inside CURRENT_DATA_DIR and reject disallowed filenames or directory escapes."""
    raw = str(path_or_name or "").strip()
    if not raw:
        raise PermissionError("ACCESS DENIED: Empty file path.")
    active_dir = CURRENT_DATA_DIR.get()
    if not active_dir:
        raise PermissionError("ACCESS DENIED: No active dataset directory configured.")
    base_dir = os.path.realpath(active_dir)
    candidate = os.path.realpath(raw if os.path.isabs(raw) else os.path.join(base_dir, raw))
    if candidate != base_dir and not candidate.startswith(base_dir + os.sep):
        raise PermissionError(f"ACCESS DENIED: '{raw}' is outside active dataset directory '{base_dir}'.")
    rel_parts = os.path.relpath(candidate, base_dir).split(os.sep)
    if any("private_do_not_release" in p.lower() for p in rel_parts):
        raise PermissionError(f"ACCESS DENIED: Private debug path '{raw}' is strictly forbidden.")
    fname = os.path.basename(candidate).lower()
    if fname not in ALLOWED_DATASET_FILENAMES:
        raise PermissionError(
            f"ACCESS DENIED: '{os.path.basename(candidate)}' is not in allowed dataset files {sorted(ALLOWED_DATASET_FILENAMES)}."
        )
    return candidate


def is_allowed_file(path_or_name: Any) -> bool:
    """Return True iff path_or_name resolves to an authorized dataset file inside CURRENT_DATA_DIR."""
    try:
        resolve_sandboxed_path(path_or_name)
        return True
    except PermissionError:
        return False


def read_allowed_dataset_file(file_path: str) -> Any:
    """Read an authorized JSON (.json -> dict/list), CSV (.csv -> DataFrame), or text file."""
    resolved = resolve_sandboxed_path(file_path)
    with builtins.open(resolved, "r", encoding="utf-8") as f:
        content = f.read()
    if resolved.lower().endswith(".json"):
        return json.loads(content)
    if resolved.lower().endswith(".csv"):
        return pd.read_csv(io.StringIO(content))
    return content


def safe_open(file_path: Any, mode: str = "r", *args: Any, **kwargs: Any):
    """Read-only open() wrapper restricted to authorized dataset files."""
    if any(m in str(mode) for m in ("w", "a", "x", "+")):
        raise PermissionError("ACCESS DENIED: Write operations are disabled inside the evaluation sandbox.")
    resolved = resolve_sandboxed_path(file_path)
    if "b" not in str(mode):
        kwargs.setdefault("encoding", "utf-8")
    return builtins.open(resolved, mode, *args, **kwargs)


class SafePath:
    """Read-only os.path proxy hiding unauthorized files."""

    def __getattr__(self, name: str) -> Any:
        if name in ("exists", "isfile"):
            return lambda p: is_allowed_file(p) and getattr(os.path, name)(resolve_sandboxed_path(p))
        return getattr(os.path, name)


class SafeOS:
    """Read-only os proxy filtering listdir() to authorized dataset filenames only."""

    path = SafePath()
    sep = os.sep
    linesep = os.linesep

    @staticmethod
    def getcwd() -> str:
        return CURRENT_DATA_DIR.get() or os.getcwd()

    @staticmethod
    def listdir(target_dir: str | None = None) -> list[str]:
        active = CURRENT_DATA_DIR.get()
        if not active:
            return []
        real_active = os.path.realpath(active)
        real_target = os.path.realpath(target_dir) if target_dir else real_active
        if real_target != real_active:
            raise PermissionError(f"ACCESS DENIED: Cannot list directory '{target_dir}'.")
        return sorted(f for f in os.listdir(real_target) if f.lower() in ALLOWED_DATASET_FILENAMES)


class SafePandas:
    """Pandas proxy restricting read_csv / read_json to authorized dataset files and blocking to_* exports."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("to_"):
            raise PermissionError(f"ACCESS DENIED: pandas.{name} is disabled in read-only sandbox.")
        if name == "read_csv":
            return lambda fp, *a, **kw: pd.read_csv(
                fp if hasattr(fp, "read") else resolve_sandboxed_path(fp), *a, **kw
            )
        if name == "read_json":
            return lambda fp, *a, **kw: pd.read_json(
                fp if hasattr(fp, "read") else resolve_sandboxed_path(fp), *a, **kw
            )
        if name.startswith("read_") or name in _PANDAS_BLOCKED:
            raise PermissionError(f"ACCESS DENIED: pandas.{name} is disabled in read-only sandbox.")
        return getattr(pd, name)


_PANDAS_BLOCKED = frozenset({"io", "ExcelWriter", "HDFStore"})
_NUMPY_BLOCKED = frozenset({
    "load", "loadtxt", "genfromtxt", "fromfile", "fromregex", "memmap", "save", "savez",
    "savez_compressed", "savetxt", "lib", "DataSource",
})


class SafeNumpy:
    """NumPy proxy without file I/O entry points."""

    def __getattr__(self, name: str) -> Any:
        if name in _NUMPY_BLOCKED:
            raise PermissionError(f"ACCESS DENIED: numpy.{name} is disabled in read-only sandbox.")
        return getattr(np, name)
