"""Google Gemini API wrapper with retries, an on-disk response cache, JSON parsing,
token accounting and a thread-pool helper for batched calls.

Authentication (in order of precedence):
  * `api_key` argument, or env `GOOGLE_API_KEY` / `GEMINI_API_KEY` (Gemini Developer API);
  * `vertexai=True` (+ `project` / `location`, or env `GOOGLE_CLOUD_PROJECT` /
    `GOOGLE_CLOUD_LOCATION`) for Vertex AI.
A `.env` file in the working directory is loaded automatically if python-dotenv is installed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

try:  # optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class ResponseCache:
    """Thread-safe sqlite key -> response-text cache."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, value))
            self._conn.commit()


# --------------------------------------------------------------------------- #
# JSON parsing
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json(text: str) -> Any:
    """Parse JSON from an LLM response, tolerating code fences and surrounding prose."""
    s = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...] span.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = s.find(open_c), s.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(s[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from response: {text[:300]!r}")


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
@dataclass
class UsageStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **kw: int) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("calls", "cache_hits", "prompt_tokens", "output_tokens", "failures")}


class GeminiClient:
    """Minimal wrapper around `google.genai.Client.models.generate_content`."""

    def __init__(
        self,
        api_key: str | None = None,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        cache_path: str | None = ".cache/llm_cache.sqlite",
        max_retries: int = 6,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        timeout_s: float | None = None,
    ):
        from google import genai

        if vertexai:
            self._client = genai.Client(
                vertexai=True,
                project=project or os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=location or os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            )
        else:
            key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("No Gemini API key: set GOOGLE_API_KEY / GEMINI_API_KEY or use vertexai=True.")
            http_options = None
            if timeout_s is not None:
                from google.genai import types

                http_options = types.HttpOptions(timeout=int(timeout_s * 1000))
            self._client = genai.Client(api_key=key, http_options=http_options)
        self.cache = ResponseCache(cache_path) if cache_path else None
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.usage = UsageStats()

    # ---- core call ------------------------------------------------------- #
    @staticmethod
    def _key(model: str, prompt: str, temperature: float, json_mode: bool, schema: dict | None,
             thinking_budget: int | None, salt: str) -> str:
        payload = json.dumps([model, prompt, temperature, json_mode, schema, thinking_budget, salt],
                             ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        json_mode: bool = False,
        schema: dict | None = None,
        max_output_tokens: int | None = None,
        thinking_budget: int | None = None,
        use_cache: bool = True,
        cache_salt: str = "",
    ) -> str:
        """Return the text of one completion. Retries with exponential backoff on any error.

        `schema` (a JSON Schema dict) enables Gemini structured output: the response is
        constrained to valid JSON matching the schema (implies json_mode).
        """
        from google.genai import types

        json_mode = json_mode or schema is not None
        key = self._key(model, prompt, temperature, json_mode, schema, thinking_budget, cache_salt)
        if use_cache and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.usage.add(cache_hits=1)
                return hit

        cfg: dict[str, Any] = {"temperature": temperature}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        if schema is not None:
            cfg["response_json_schema"] = schema
        if max_output_tokens is not None:
            cfg["max_output_tokens"] = max_output_tokens
        if thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
        config = types.GenerateContentConfig(**cfg)

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(model=model, contents=prompt, config=config)
                text = resp.text
                if text is None:
                    raise RuntimeError(f"Empty response (finish info: {getattr(resp, 'candidates', None)!r:.300})")
                um = getattr(resp, "usage_metadata", None)
                self.usage.add(
                    calls=1,
                    prompt_tokens=int(getattr(um, "prompt_token_count", 0) or 0),
                    output_tokens=int(getattr(um, "candidates_token_count", 0) or 0),
                )
                if use_cache and self.cache is not None:
                    self.cache.set(key, text)
                return text
            except Exception as e:  # noqa: BLE001 - API errors are heterogeneous
                last_err = e
                if attempt == self.max_retries:
                    break
                delay = min(self.max_delay, self.base_delay * 2**attempt) * (0.5 + random.random())
                logger.warning("Gemini call failed (%s: %s); retry %d in %.1fs", type(e).__name__, e, attempt + 1, delay)
                time.sleep(delay)
        self.usage.add(failures=1)
        raise RuntimeError(f"Gemini call failed after {self.max_retries + 1} attempts") from last_err

    def generate_json(
        self,
        prompt: str,
        model: str,
        schema: dict | None = None,
        temperature: float = 0.0,
        parse_retries: int = 2,
        validator: Callable[[Any], bool] | None = None,
        **kw: Any,
    ) -> Any:
        """Generate and parse a JSON response, constrained by `schema` when given.
        Re-samples (bypassing the cache entry) on invalid JSON or when `validator(obj)` is False."""
        last: Exception | None = None
        for k in range(parse_retries + 1):
            text = self.generate(prompt, model, temperature=temperature, json_mode=True, schema=schema,
                                 cache_salt=f"try{k}" if k else "", **kw)
            try:
                obj = parse_json(text)
                if validator is not None and not validator(obj):
                    raise ValueError(f"Response failed validation: {text[:300]!r}")
                return obj
            except ValueError as e:
                last = e
        raise ValueError(f"No valid JSON after {parse_retries + 1} tries") from last


# --------------------------------------------------------------------------- #
# Batched execution
# --------------------------------------------------------------------------- #
def thread_map(
    fn: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    max_workers: int = 16,
    desc: str | None = None,
    show_progress: bool = True,
) -> list[R]:
    """Order-preserving parallel map (I/O-bound LLM calls)."""
    items = list(items)
    if not items:
        return []
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(fn, items)
        if tqdm is not None and show_progress:
            it = tqdm(it, total=len(items), desc=desc)
        return list(it)
