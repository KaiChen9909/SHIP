"""Multi-turn sandboxed Python REPL code agent built on the google-genai SDK.

The agent gets one tool, `execute_python_code` (a persistent per-session REPL over the dataset
directory, see `sandbox.py`), plus `finish_run` for structured outputs. Report-mode sessions
(`task_type="aggregated_analysis"`) end with a Markdown report written as plain text.

Several sessions may run concurrently (asyncio + worker threads); stdout/stderr of agent code
is captured per call through a context-local router instead of process-global redirection.
"""

from __future__ import annotations

import asyncio
import builtins as _builtins
import collections
import contextvars
import io
import json
import logging
import math
import re
import sys
import threading
import time
import traceback
from typing import Any

from google.genai import types

from utils.llm import GeminiClient

from .agent_functions import get_helper_functions_prompt_doc, tabular_crosstab, tabular_stat, tree_inspect
from .dataset_manager import DatasetContext
from .sandbox import CURRENT_DATA_DIR, SafeNumpy, SafeOS, SafePandas, read_allowed_dataset_file, safe_open

logger = logging.getLogger(__name__)

DEFAULT_EVAL_MODEL = "gemini-3.5-flash-lite"  # paper: QA agent built on Gemini-3.5-Flash-Lite
_MAX_STDOUT_CHARS = 15000


# --------------------------------------------------------------------------- #
# Per-call stdout / stderr capture (safe under concurrent threads)
# --------------------------------------------------------------------------- #
_CAPTURE: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar("agent_capture", default=None)


class _StreamRouter(io.TextIOBase):
    """Writes to the active capture buffer of the current context, else to the real stream."""

    def __init__(self, real):
        self._real = real

    def write(self, s: str) -> int:
        buf = _CAPTURE.get()
        return buf.write(s) if buf is not None else self._real.write(s)

    def flush(self) -> None:
        if _CAPTURE.get() is None:
            self._real.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


_ROUTER_LOCK = threading.Lock()


def _install_stream_router() -> None:
    with _ROUTER_LOCK:
        if not isinstance(sys.stdout, _StreamRouter):
            sys.stdout = _StreamRouter(sys.stdout)
        if not isinstance(sys.stderr, _StreamRouter):
            sys.stderr = _StreamRouter(sys.stderr)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def extract_json_dict(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from raw model text or fenced json blocks."""
    s = (text or "").strip()
    if not s:
        return None
    candidates = [s]
    fence = "`" * 3
    m = re.search(rf"{fence}(?:json)?\s*(\{{.*?\}})\s*{fence}", s, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        candidates.append(s[start : end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict):
            return obj
    return None


_ALLOWED_BUILTINS = (
    "abs", "all", "any", "ascii", "bin", "bool", "bytes", "callable", "chr", "complex", "dict",
    "dir", "divmod", "enumerate", "filter", "float", "format", "frozenset", "getattr", "hasattr",
    "hash", "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "object", "oct", "ord", "pow", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "type", "zip", "None", "True", "False",
    "Exception", "ArithmeticError", "AssertionError", "AttributeError", "IndexError", "KeyError",
    "LookupError", "NameError", "NotImplementedError", "RuntimeError", "StopIteration",
    "TypeError", "ValueError", "ZeroDivisionError", "PermissionError",
)
_ALLOWED_IMPORTS = frozenset({
    "collections", "csv", "datetime", "functools", "itertools", "json", "math", "numpy",
    "operator", "pandas", "re", "scipy", "statistics", "string", "textwrap", "typing",
})


def builtins_whitelist() -> dict[str, Any]:
    """Builtins exposed inside the evaluation REPL (see sandbox.py for the threat model)."""
    safe = {k: getattr(_builtins, k) for k in _ALLOWED_BUILTINS if hasattr(_builtins, k)}
    orig_import = _builtins.__import__

    def _safe_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0):
        root = name.split(".")[0]
        if root == "os":
            return SafeOS()
        if root == "pandas":
            return SafePandas()
        if root == "numpy":
            return SafeNumpy()
        if root not in _ALLOWED_IMPORTS or name.startswith("scipy.io"):
            raise ImportError(f"Import of '{name}' is blocked inside the read-only evaluation sandbox.")
        return orig_import(name, globals_, locals_, fromlist, level)

    safe["__import__"] = _safe_import
    safe["open"] = safe_open
    return safe


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class CodeAgentExecutor:
    """Multi-turn Python REPL agent sessions plus single-turn LLM calls (splitting, judging)."""

    def __init__(
        self,
        model_name: str = DEFAULT_EVAL_MODEL,
        api_key: str | None = None,
        vertexai: bool = False,
        max_iters: int = 30,
        exec_timeout_s: float = 300.0,
        max_retries: int = 5,
        cache_path: str | None = None,
    ):
        self.model_name = model_name
        self.max_iters = max_iters
        self.exec_timeout_s = exec_timeout_s
        self.max_retries = max_retries
        self.llm = GeminiClient(api_key=api_key, vertexai=vertexai, cache_path=cache_path, max_retries=max_retries)
        self.client = self.llm.client
        _install_stream_router()

    @property
    def usage(self) -> dict[str, int]:
        return self.llm.usage.to_dict()

    async def llm_call(
        self, prompt: str, temperature: float = 0.0, cache_salt: str = "", schema: dict | None = None
    ) -> str:
        """Single-turn generation without tools (statement splitting, QA judge).
        `schema` enables Gemini structured output (response constrained to that JSON Schema)."""
        return await asyncio.to_thread(
            self.llm.generate, prompt, self.model_name, temperature=temperature, cache_salt=cache_salt, schema=schema
        )

    # ---- REPL ------------------------------------------------------------ #
    def _build_repl_globals(self, dataset: DatasetContext, task_type: str) -> dict[str, Any]:
        csv_path = (
            dataset.full_path_csv_path
            if task_type in ("aggregated_analysis", "statement_verification")
            else dataset.main_csv_path
        )
        return {
            "__builtins__": builtins_whitelist(),
            "pd": SafePandas(),
            "np": SafeNumpy(),
            "json": json,
            "re": re,
            "math": math,
            "collections": collections,
            "os": SafeOS(),
            "open": safe_open,
            "read_allowed_dataset_file": read_allowed_dataset_file,
            "tree_inspect": tree_inspect,
            "tabular_stat": tabular_stat,
            "tabular_crosstab": tabular_crosstab,
            "data_dir": dataset.data_dir,
            "data_file_path": csv_path,
            "category_tree_path": dataset.category_tree_path,
        }

    @staticmethod
    def _exec_in_sandbox(code: str, data_dir: str, session_globals: dict[str, Any]) -> str:
        buf = io.StringIO()
        dir_token = CURRENT_DATA_DIR.set(data_dir)
        cap_token = _CAPTURE.set(buf)
        try:
            exec(compile(code, "<agent_repl>", "exec"), session_globals, session_globals)  # noqa: S102
        except Exception:  # noqa: BLE001
            buf.write("\n[STDERR]\n" + traceback.format_exc())
        finally:
            _CAPTURE.reset(cap_token)
            CURRENT_DATA_DIR.reset(dir_token)
        out = buf.getvalue()
        if not out.strip():
            out = "(Code executed successfully with no stdout output. Use print(...) to inspect values.)"
        if len(out) > _MAX_STDOUT_CHARS:
            half = _MAX_STDOUT_CHARS // 2
            out = out[:half] + f"\n... [truncated {len(out) - _MAX_STDOUT_CHARS} chars] ...\n" + out[-half:]
        return out

    async def _run_code(self, code: str, dataset: DatasetContext, session_globals: dict[str, Any]) -> str:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._exec_in_sandbox, code, dataset.data_dir, session_globals),
                timeout=self.exec_timeout_s,
            )
        except asyncio.TimeoutError:
            # The worker thread cannot be killed; it is abandoned and the session continues.
            return f"[STDERR]\nExecution exceeded {self.exec_timeout_s:.0f}s and was abandoned. Write faster code."

    async def _generate(self, contents, cfg, turn_idx: int) -> tuple[Any, float]:
        slept = 0.0
        for attempt in range(self.max_retries):
            try:
                resp = await asyncio.to_thread(
                    self.client.models.generate_content, model=self.model_name, contents=contents, config=cfg
                )
                um = getattr(resp, "usage_metadata", None)
                self.llm.usage.add(
                    calls=1,
                    prompt_tokens=int(getattr(um, "prompt_token_count", 0) or 0),
                    output_tokens=int(getattr(um, "candidates_token_count", 0) or 0),
                )
                return resp, slept
            except Exception as e:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    self.llm.usage.add(failures=1)
                    raise
                wait = min(60.0, 2.0 * 2**attempt)
                slept += wait
                logger.warning("Agent turn %d retry %d/%d after %.0fs: %s", turn_idx, attempt + 1, self.max_retries, wait, e)
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable")

    async def run_agent_query(
        self,
        dataset: DatasetContext,
        prompt: str,
        algo_name: str | None = None,
        temperature: float = 0.0,
        task_type: str = "answer",
    ) -> tuple[str, list[dict[str, Any]]]:
        """Run one multi-turn REPL session on `dataset`. Returns (final output, trace events).

        task_type: "aggregated_analysis" (Markdown report, no finish_run), or "question_generation"
        / "answer" / "statement_verification" (JSON output via finish_run or final text).
        """
        session_globals = self._build_repl_globals(dataset, task_type)
        trace: list[dict[str, Any]] = []
        retry_sleep = 0.0
        report_mode = task_type == "aggregated_analysis"

        decls = [
            types.FunctionDeclaration(
                name="execute_python_code",
                description=(
                    "Execute Python 3 code in a persistent read-only session with pre-loaded `pd`, `np`, `json`, "
                    "`read_allowed_dataset_file`, `tree_inspect`, `tabular_stat`, `tabular_crosstab`, "
                    "`data_dir`, `data_file_path`, and `category_tree_path`. Always print(...) results to STDOUT."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python code to execute."}},
                    "required": ["code"],
                },
            )
        ]
        if not report_mode:
            decls.append(
                types.FunctionDeclaration(
                    name="finish_run",
                    description="Submit the final structured JSON dictionary output and end the session.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "output": {"type": "object", "description": "Final JSON dictionary mapping IDs to results."}
                        },
                        "required": ["output"],
                    },
                )
            )

        system_instruction = (
            "You are an expert empirical data analyst operating a stateful Python REPL.\n"
            f"Active dataset directory: {dataset.data_dir}\n"
            f"Primary CSV file (`data_file_path`): {session_globals['data_file_path']}\n"
            f"Taxonomy tree file (`category_tree_path`): {dataset.category_tree_path}\n"
            + get_helper_functions_prompt_doc()
            + "\nNote: in SHIP outputs (`synthetic.csv`), each attribute `<dim>` has divided level columns "
            "`'<dim> (level 1)'`, `'<dim> (level 2)'`, ... and a full path column `'<dim> (path)'`; "
            "`taxonomy.json` / `category_tree.json` work with `tree_inspect`, `tabular_stat`, and `tabular_crosstab`."
        )
        cfg = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            tools=[types.Tool(function_declarations=decls)],
        )
        contents: list[types.Content] = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

        last_text = ""
        finished: dict[str, Any] | None = None
        for turn_idx in range(1, self.max_iters + 1):
            resp, slept = await self._generate(contents, cfg, turn_idx)
            retry_sleep += slept
            if not resp.candidates or not resp.candidates[0].content:
                break
            model_content = resp.candidates[0].content
            contents.append(model_content)
            parts = model_content.parts or []
            fn_calls = [p.function_call for p in parts if p.function_call]
            turn_text = "\n".join(p.text for p in parts if p.text and not getattr(p, "thought", False)).strip()
            if turn_text:
                last_text = turn_text
                trace.append({"event_type": "turn_summary", "turn_index": turn_idx, "llm_reasoning": turn_text[:1000]})
            if not fn_calls:
                break  # final response

            responses: list[types.Part] = []
            for fc in fn_calls:
                args = dict(fc.args or {})
                if fc.name == "finish_run":
                    out = args.get("output", args)
                    if isinstance(out, str):
                        out = extract_json_dict(out) or {"raw": out}
                    finished = out if isinstance(out, dict) else {"output": out}
                    trace.append({"event_type": "finish_run", "final_output": finished})
                    responses.append(types.Part.from_function_response(name="finish_run", response={"status": "completed"}))
                elif fc.name == "execute_python_code":
                    code = str(args.get("code", ""))
                    t0 = time.time()
                    out = await self._run_code(code, dataset, session_globals)
                    trace.append({
                        "event_type": "tool_execution",
                        "tool_name": "execute_python_code",
                        "code_written": code,
                        "execution_output": out[:2000],
                        "duration_seconds": round(time.time() - t0, 3),
                    })
                    responses.append(types.Part.from_function_response(name="execute_python_code", response={"output": out}))
                else:
                    responses.append(types.Part.from_function_response(
                        name=fc.name or "unknown", response={"error": f"Unknown tool: {fc.name}"}
                    ))
            if finished is not None:
                break
            if turn_idx == self.max_iters - 1:
                wrap = (
                    "FINAL TURN WARNING: Stop calling `execute_python_code` now and write out the complete final Markdown report."
                    if report_mode
                    else "FINAL TURN WARNING: Stop calling `execute_python_code` now and return the complete JSON "
                    "dictionary (via `finish_run` or plain JSON text)."
                )
                responses.append(types.Part.from_text(text=wrap))
            contents.append(types.Content(role="user", parts=responses))

        trace.append({"event_type": "timing_summary", "retry_sleep_seconds": round(retry_sleep, 3)})
        if report_mode:
            return last_text, trace
        if finished is not None:
            return json.dumps(finished, ensure_ascii=False), trace
        extracted = extract_json_dict(last_text)
        return (json.dumps(extracted, ensure_ascii=False) if extracted is not None else last_text), trace
