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


class QAAnswerer:
    """Answers analytical question banks on a target DatasetContext using partitioned type batches."""

    def __init__(self, agent_executor: CodeAgentExecutor):
        self.agent = agent_executor

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
                q_block = "\n".join(
                    f"- [{q.get('id')}] (Type: {q_type}, Scope: {q.get('attr_type', 'single_attr')}): {q.get('question', '')}"
                    for q in q_batch
                )
                prompt = safe_format_prompt(
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

                t0 = time.time()
                for attempt in range(3):
                    try:
                        raw, agent_trace = await self.agent.run_agent_query(
                            target_dataset,
                            prompt,
                            algo_name=algo_name or target_dataset.algo_name,
                            temperature=0.0,
                            task_type="answer",
                        )
                        parsed = extract_json_dict(raw) or {}
                        for wrap_k in ("answers", "results", "output", "data"):
                            if wrap_k in parsed and isinstance(parsed[wrap_k], dict):
                                parsed = parsed[wrap_k]
                                break

                        batch_map: dict[str, dict[str, Any]] = {}
                        by_norm = {_norm_id(kp): vp for kp, vp in parsed.items()}
                        for q in q_batch:
                            qid = str(q.get("id", ""))
                            ans = parsed.get(qid)
                            if ans is None:
                                ans = by_norm.get(_norm_id(qid))
                            if isinstance(ans, dict):
                                val = ans.get("answer_value", ans.get("answer"))
                                disp = str(ans.get("display_answer", val))
                                expl = str(ans.get("explanation", ""))
                            elif ans is not None:
                                val, disp, expl = ans, str(ans), str(ans)
                            else:
                                val, disp, expl = None, "N/A", "Missing structured answer."
                            batch_map[qid] = {
                                "id": qid,
                                "question": str(q.get("question", "")),
                                "question_type": q_type,
                                "answer_value": val,
                                "display_answer": disp,
                                "explanation": expl,
                            }

                        if any(v["answer_value"] is not None for v in batch_map.values()) or attempt == 2:
                            dur = round(time.time() - t0, 3)
                            tool_dur = round(
                                sum(e.get("duration_seconds", 0.0) for e in agent_trace if e.get("event_type") == "tool_execution"),
                                3,
                            )
                            return batch_map, {
                                "batch_index": b_idx,
                                "question_type": q_type,
                                "questions_count": len(q_batch),
                                "duration_seconds": dur,
                                "tool_execution_seconds": tool_dur,
                                "tool_call_count": sum(1 for e in agent_trace if e.get("event_type") == "tool_execution"),
                                "trace": agent_trace,
                            }
                        await asyncio.sleep(3.0 * (attempt + 1))
                    except Exception as e:  # noqa: BLE001
                        if attempt == 2:
                            logger.warning("Answer batch %d (%s) failed: %s", b_idx, q_type, e)
                            return {
                                str(q.get("id", "")): {
                                    "id": str(q.get("id", "")),
                                    "question": str(q.get("question", "")),
                                    "question_type": q_type,
                                    "answer_value": None,
                                    "display_answer": "Error",
                                    "explanation": str(e),
                                }
                                for q in q_batch
                            }, {"batch_index": b_idx, "question_type": q_type, "error": str(e), "trace": []}
                        await asyncio.sleep(3.0 * (attempt + 1))
                return {}, {"batch_index": b_idx, "trace": []}

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
