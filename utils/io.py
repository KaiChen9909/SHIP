"""Data loading, prompt rendering and small text helpers."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import pandas as pd


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def _render_turns(turns: list) -> str:
    """Render a list of chat turns ({'role','content'} dicts or strings) as plain text."""
    lines = []
    for t in turns:
        if isinstance(t, dict):
            role = t.get("role") or t.get("speaker") or t.get("from") or "turn"
            content = t.get("content") or t.get("text") or t.get("value") or ""
            if role in ("assistant", "model", "gpt", "bot"):
                role = "ai"
            lines.append(f"{role}: {content}")
        else:
            lines.append(str(t))
    return "\n".join(lines)


def record_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _render_turns(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def load_records(path: str, text_field: str = "text", max_records: int | None = None) -> list[str]:
    """Load textual records from .jsonl / .json / .csv / .parquet / .txt (one record per line).

    `text_field` may hold a string or a list of chat turns.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f if line.strip()]
    else:
        if ext == ".jsonl":
            df = pd.read_json(path, lines=True)
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported data format: {ext}")
        if text_field not in df.columns:
            raise KeyError(f"Field {text_field!r} not in {list(df.columns)}")
        texts = [record_to_text(v) for v in df[text_field].tolist()]
    return texts[:max_records] if max_records else texts


def read_text_or_literal(value: str | None) -> str | None:
    """Return file contents if `value` is an existing path, otherwise `value` itself."""
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as f:
            return f.read().strip()
    return value


def load_workload(value: str) -> list[str]:
    """Workload from a .json list, a .txt file (one query per line) or a literal string."""
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as f:
            raw = f.read()
        if value.endswith(".json"):
            obj = json.loads(raw)
            return [str(q) for q in (obj if isinstance(obj, list) else obj["queries"])]
        return [line.strip() for line in raw.splitlines() if line.strip()]
    return [value]


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _load_template(path: str):
    import jinja2

    with open(path, encoding="utf-8") as f:
        src = f.read()
    if not src.strip():
        raise ValueError(f"Prompt template {path} is empty; please fill it in.")
    env = jinja2.Environment(
        undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False
    )
    return env.from_string(src)


def render_prompt(path: str, **variables: Any) -> str:
    """Render a Jinja2 prompt template file."""
    return _load_template(path).render(**variables).strip()


def truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"
