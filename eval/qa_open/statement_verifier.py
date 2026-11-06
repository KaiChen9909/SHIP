"""Parallel multi-agent verifier scoring atomic statements against a recipient DatasetContext."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from eval.common.code_agent import CodeAgentExecutor, extract_json_dict
from eval.common.dataset_manager import DatasetContext, DatasetTemplateManager, safe_format_prompt

logger = logging.getLogger(__name__)


class StatementVerifier:
    """Validates batches of atomic statements on a recipient dataset using Python REPL verification."""

    def __init__(self, agent_executor: CodeAgentExecutor):
        self.agent = agent_executor

    async def _verify_batch(
        self,
        recipient_dataset: DatasetContext,
        batch_stmts: list[dict[str, Any]],
        template_str: str,
        workload_content: str,
        min_supported_score: int,
        algo_name: str | None,
        workload_title: str,
        workload_description: str,
        workload_key: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        lines = [
            f"{stmt.get('id', f'statement {idx + 1}')}: {stmt.get('statement', '')}"
            for idx, stmt in enumerate(batch_stmts)
        ]
        prompt = safe_format_prompt(
            template_str,
            dataset_description=recipient_dataset.intro_text or "",
            statements_file="\n".join(lines),
            data_dir=recipient_dataset.data_dir,
            workload_content=workload_content,
            workload_title=workload_title,
            workload_description=workload_description,
            data_file_path=recipient_dataset.full_path_csv_path,
            category_tree_path=recipient_dataset.category_tree_path,
            dataset_info_path=recipient_dataset.dataset_info_path,
            k=len(batch_stmts),
            num_statements=len(batch_stmts),
        )

        raw, trace = await self.agent.run_agent_query(
            recipient_dataset,
            prompt,
            algo_name=algo_name or recipient_dataset.algo_name,
            temperature=0.0,
            task_type="statement_verification",
        )
        parsed = extract_json_dict(raw) or {}
        if not parsed:
            for num_str, sc_str in re.findall(
                r"(?i)(?:[\*\-\s\d\.]*)\b(?:statement\s+)?(\d+)\b[\*\s]*[:\-]\s*([0-5](?:\.\d+)?)", raw
            ):
                parsed[f"statement {num_str}"] = float(sc_str)

        for wrap_k in ("output", "val", "statement_scores", "scores", "results", "verifications"):
            if wrap_k in parsed and isinstance(parsed[wrap_k], dict):
                parsed = parsed[wrap_k]
                break

        norm_scores = {
            str(k).strip(' "\'').replace("_", " ").lower().strip(): v for k, v in parsed.items()
        }

        verifications: list[dict[str, Any]] = []
        for idx, item in enumerate(batch_stmts):
            orig_id = str(item.get("id", f"statement {idx + 1}"))
            clean_orig = orig_id.strip(' "\'').replace("_", " ").lower().strip()
            clean_local = f"statement {idx + 1}"
            score = norm_scores.get(clean_orig, norm_scores.get(clean_local, norm_scores.get(str(idx + 1))))
            if score is None:
                m = re.search(r"\d+", orig_id)
                if m:
                    score = norm_scores.get(f"statement {m.group(0)}", norm_scores.get(m.group(0)))
            try:
                score_f = float(score) if score is not None else 0.0
            except Exception:  # noqa: BLE001
                score_f = 0.0
            verifications.append({
                "statement_id": orig_id,
                "statement": item.get("statement", ""),
                "score": score_f,
                "supported": score_f >= min_supported_score,
                "workload_key": item.get("workload_key") or workload_key,
                "workload_title": item.get("workload_title") or workload_title,
            })
        return verifications, trace

    async def verify(
        self,
        recipient_dataset: DatasetContext,
        statements_path: str,
        workload_content: str,
        prompt_template_path: str,
        min_supported_score: int = 4,
        batch_size: int = 10,
        parallel_num: int = 5,
        algo_name: str | None = None,
        workload_title: str = "",
        workload_description: str = "",
        workload_key: str = "",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load statements JSON and verify in parallel batches of `batch_size`."""
        with open(statements_path, encoding="utf-8") as f:
            data = json.load(f)
        statements = (
            data if isinstance(data, list)
            else [{"id": str(k), "statement": str(v)} for k, v in data.items() if isinstance(v, str)]
        )
        if not statements:
            return [], []

        template_str = DatasetTemplateManager.load_prompt_template(prompt_template_path)
        step = max(1, batch_size)
        batches = [statements[i : i + step] for i in range(0, len(statements), step)]
        sem = asyncio.Semaphore(max(1, min(parallel_num, len(batches))))

        async def _guarded(b: list[dict[str, Any]]):
            async with sem:
                return await self._verify_batch(
                    recipient_dataset,
                    b,
                    template_str,
                    workload_content,
                    min_supported_score,
                    algo_name,
                    workload_title,
                    workload_description,
                    workload_key,
                )

        results = await asyncio.gather(*[_guarded(b) for b in batches], return_exceptions=True)
        all_vers: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        for idx, res in enumerate(results):
            if isinstance(res, BaseException):
                logger.error("Statement verification batch %d failed: %s", idx + 1, res)
                for s in batches[idx]:
                    all_vers.append({
                        "statement_id": s.get("id", "unknown"),
                        "statement": s.get("statement", ""),
                        "score": 0.0,
                        "supported": False,
                    })
                continue
            b_vers, b_trace = res
            all_vers.extend(b_vers)
            if idx < 5 and b_trace:
                traces.append({"statement_batch_idx": idx + 1, "trace_events": b_trace})
        return all_vers, traces
