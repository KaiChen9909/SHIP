"""Batched async QA answerer over target hierarchical SHIP datasets."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

from eval.common.code_agent import CodeAgentExecutor, extract_json_dict
from eval.common.dataset_manager import DatasetContext, DatasetTemplateManager, safe_format_prompt

logger = logging.getLogger(__name__)

_HIE_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts", "hie_table")


def _norm_id(key: Any) -> str:
    """Normalize an answer key such as '[single_attr_yes_no_1]' or 'Single Attr Yes No 1'.
    Exact (normalized) matching only: substring matching would confuse '..._1' with '..._10'."""
    return re.sub(r"[\s_\-\[\]()'\"]+", "_", str(key).strip().lower()).strip("_")


def _answer_prompt_file(question_type: str) -> str:
    q = question_type.lower()
    if any(k in q for k in ("compare", "comparison", "contrast")):
        return "answer_compare_prompt.json"
    if any(k in q for k in ("ask_for_number", "number", "numerical", "percentage", "proportion")):
        return "answer_ask_for_number_prompt.json"
    return "answer_yes_no_prompt.json"


_POSITIONAL_KEY = re.compile(r"^(?:q|question|id)?_?(\d+)$")


def match_answers(parsed: dict[str, Any], questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Map the agent's answer keys to question IDs.

    1. exact / normalized ID match ('[single_attr_yes_no_1]', 'Single Attr Yes No 1');
    2. positional keys ('question 3', 'q3', '3') -> the 3rd question of the batch;
    3. if no key matched at all and the agent returned exactly one answer per question,
       the answers are taken in order.
    """
    for wrap_k in ("answers", "results", "output", "data"):
        if wrap_k in parsed and isinstance(parsed[wrap_k], dict):
            parsed = parsed[wrap_k]
            break
    ids = [str(q.get("id", "")) for q in questions]
    by_norm = {_norm_id(k): v for k, v in parsed.items()}
    out: dict[str, Any] = {}
    for qid in ids:
        ans = parsed.get(qid, by_norm.get(_norm_id(qid)))
        if ans is not None:
            out[qid] = ans
    id_norms = {_norm_id(q) for q in ids}
    for k, v in parsed.items():
        m = _POSITIONAL_KEY.match(_norm_id(k))
        if m and _norm_id(k) not in id_norms and 1 <= int(m.group(1)) <= len(ids):
            out.setdefault(ids[int(m.group(1)) - 1], v)
    if not out and len(parsed) == len(ids):
        out = dict(zip(ids, parsed.values()))
    return {k: v for k, v in out.items() if v is not None}


def _answer_record(q: dict[str, Any], q_type: str, ans: Any, error: str = "") -> dict[str, Any]:
    if isinstance(ans, dict):
        val = ans.get("answer_value", ans.get("answer"))
        disp, expl = str(ans.get("display_answer", val)), str(ans.get("explanation", ""))
    elif ans is not None:
        val, disp, expl = ans, str(ans), str(ans)
    else:
        val, disp, expl = None, "N/A", f"Missing structured answer. {error}".strip()
    return {
        "id": str(q.get("id", "")),
        "question": str(q.get("question", "")),
        "question_type": q_type,
        "answer_value": val,
        "display_answer": disp,
        "explanation": expl,
    }


class QAAnswerer:
    """Answers analytical question banks on a target DatasetContext using partitioned type batches."""

    def __init__(self, agent_executor: CodeAgentExecutor, max_attempts: int = 3):
        self.agent = agent_executor
        self.max_attempts = max_attempts

    @staticmethod
    def _build_prompt(template_str, target_dataset, q_type, q_batch, workload_title, workload_description) -> str:
        q_block = "\n".join(
            f"- [{q.get('id')}] (Type: {q_type}, Scope: {q.get('attr_type', 'single_attr')}): {q.get('question', '')}"
            for q in q_batch
        )
        return safe_format_prompt(
            template_str,
            dataset_description=target_dataset.intro_text or target_dataset.algo_name,
            workload_title=workload_title,
            workload_description=workload_description,
            data_dir=target_dataset.data_dir,
            data_file_path=target_dataset.main_csv_path,
            category_tree_path=target_dataset.category_tree_path,
            dataset_info_path=target_dataset.dataset_info_path,
            algo_name=target_dataset.algo_name,
            data_files=", ".join(target_dataset.data_files),
            questions_block=q_block,
        )

    async def answer_questions(
        self,
        target_dataset: DatasetContext,
        questions: list[dict[str, Any]],
        algo_name: str | None = None,
        workload_title: str = "",
        workload_description: str = "",
        batch_size: int = 10,
        parallel_num: int = 10,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        """Partition questions by question_type and answer in parallel batches."""
        by_type: dict[str, list[dict[str, Any]]] = {}
        for q in questions:
            by_type.setdefault(str(q.get("question_type", "yes_no")).lower(), []).append(q)

        sem = asyncio.Semaphore(max(1, parallel_num))

        async def _run_batch(q_type: str, q_batch: list[dict[str, Any]], b_idx: int):
            async with sem:
                template_str = DatasetTemplateManager.load_prompt_template(
                    os.path.join(_HIE_PROMPTS_DIR, _answer_prompt_file(q_type))
                )
                batch_map: dict[str, dict[str, Any]] = {}
                traces: list[dict[str, Any]] = []
                errors: list[str] = []
                pending = list(q_batch)
                t0 = time.time()
                # Only questions still missing an answer are re-asked, so one bad key never drops a batch.
                for attempt in range(self.max_attempts):
                    if attempt:
                        await asyncio.sleep(3.0 * attempt)
                    try:
                        raw, agent_trace = await self.agent.run_agent_query(
                            target_dataset,
                            self._build_prompt(template_str, target_dataset, q_type, pending,
                                               workload_title, workload_description),
                            algo_name=algo_name or target_dataset.algo_name,
                            temperature=0.0,
                            task_type="answer",
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Answer batch %d (%s) attempt %d failed: %s", b_idx, q_type, attempt + 1, e)
                        errors.append(str(e))
                        continue
                    traces.extend(agent_trace)
                    by_id = {str(q.get("id", "")): q for q in pending}
                    for qid, ans in match_answers(extract_json_dict(raw) or {}, pending).items():
                        rec = _answer_record(by_id[qid], q_type, ans)
                        if rec["answer_value"] is not None:
                            batch_map[qid] = rec
                    pending = [q for q in pending if str(q.get("id", "")) not in batch_map]
                    if not pending:
                        break

                for q in pending:
                    logger.warning("No answer for question %s after %d attempts", q.get("id"), self.max_attempts)
                    batch_map[str(q.get("id", ""))] = _answer_record(q, q_type, None, error="; ".join(errors[-1:]))
                tool_events = [e for e in traces if e.get("event_type") == "tool_execution"]
                return batch_map, {
                    "batch_index": b_idx,
                    "question_type": q_type,
                    "questions_count": len(q_batch),
                    "unanswered_count": len(pending),
                    "duration_seconds": round(time.time() - t0, 3),
                    "tool_execution_seconds": round(sum(e.get("duration_seconds", 0.0) for e in tool_events), 3),
                    "tool_call_count": len(tool_events),
                    "errors": errors,
                    "trace": traces,
                }

        tasks = []
        b_idx = 0
        step = max(1, batch_size)
        for q_type, q_list in by_type.items():
            for i in range(0, len(q_list), step):
                tasks.append(_run_batch(q_type, q_list[i : i + step], b_idx))
                b_idx += 1

        results = await asyncio.gather(*tasks)
        answers_map: dict[str, dict[str, Any]] = {}
        traces: list[dict[str, Any]] = []
        for b_map, b_trace in results:
            answers_map.update(b_map)
            traces.append(b_trace)
        return answers_map, traces
