"""End-to-end Closed-Ended Direct QA benchmark orchestrator."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from typing import Any, Sequence

from eval.common.code_agent import DEFAULT_EVAL_MODEL, CodeAgentExecutor
from eval.common.dataset_manager import DatasetTemplateManager

from .qa_answerer import QAAnswerer
from .qa_scorer import QAScorer
from .question_generator import QuestionGenerator


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    return m, math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def _summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [q["score"] for q in items]
    errs = [q["absolute_error"] for q in items if q.get("absolute_error") is not None]
    n = len(scores)
    return {
        "count": n,
        "average_score": round(sum(scores) / n, 4) if n else 0.0,
        "accuracy_t4": round(sum(1 for s in scores if s >= 4) / n, 4) if n else 0.0,
        "accuracy_t5": round(sum(1 for s in scores if s == 5) / n, 4) if n else 0.0,
        "mae": round(sum(errs) / len(errs), 4) if errs else None,
    }


class QueryAnswerEvaluator:
    """Runs GT question generation -> Target SHIP answering -> 1–5 rubric scoring."""

    def __init__(
        self,
        model_name: str = DEFAULT_EVAL_MODEL,
        vertexai: bool = False,
        cache_dir: str = ".cache/eval_questions",
    ):
        self.model_name = model_name
        self.agent = CodeAgentExecutor(model_name=model_name, vertexai=vertexai)
        self.question_generator = QuestionGenerator(self.agent, cache_dir=cache_dir)
        self.answerer = QAAnswerer(self.agent)
        self.scorer = QAScorer(self.agent)

    async def evaluate_async(
        self,
        target_data_dir: str,
        workload_file_path: str,
        gt_data_dir: str,
        target_algo_name: str = "ship",
        gt_algo_name: str = "groundtruth",
        dataset_name: str = "wildchat",
        dataset_description: str = "",
        num_rounds: int = 1,
        questions_per_type: int = 50,
        attr_types: Sequence[str] = ("single_attr", "multi_attr"),
        question_types: Sequence[str] = ("yes_no", "compare", "ask_for_number"),
        batch_size: int = 10,
        parallel_num: int = 10,
        question_bank_path: str | None = None,
        output_report_path: str | None = None,
        overwrite_cache: bool = False,
        max_questions: int | None = None,
    ) -> dict[str, Any]:
        """Run Closed-Ended QA evaluation and write JSON report."""
        t0 = time.time()
        gt_ds = DatasetTemplateManager.load_dataset(gt_algo_name, gt_data_dir, dataset_name, dataset_description)
        tgt_ds = DatasetTemplateManager.load_dataset(target_algo_name, target_data_dir, dataset_name, dataset_description)
        workload_dict = DatasetTemplateManager.load_workload_dict(workload_file_path)
        wl_id = os.path.splitext(os.path.basename(workload_file_path))[0]
        report_path = output_report_path or os.path.join(tgt_ds.data_dir, f"qa_eval_report_{wl_id}.json")

        rounds_summary: list[dict[str, Any]] = []
        all_scored: list[dict[str, Any]] = []

        for r_idx in range(1, num_rounds + 1):
            round_scored: list[dict[str, Any]] = []
            for wl_key, wl_item in workload_dict.items():
                wl_title, wl_desc = wl_item.get("title", ""), wl_item.get("description", "")
                wl_name = f"{wl_id}_wl{wl_key}_r{r_idx}" if num_rounds > 1 else f"{wl_id}_wl{wl_key}"

                if question_bank_path and os.path.isfile(question_bank_path):
                    with open(question_bank_path, encoding="utf-8") as f:
                        questions = json.load(f)
                    questions = [
                        q for q in questions
                        if q.get("question_type") in set(question_types) and q.get("attr_type") in set(attr_types)
                    ]
                else:
                    _, questions = await self.question_generator.generate_question_bank(
                        dataset=gt_ds,
                        workload_content=json.dumps(wl_item),
                        workload_name=wl_name,
                        workload_title=wl_title,
                        workload_description=wl_desc,
                        workload_key=str(wl_key),
                        attr_types=attr_types,
                        question_types=question_types,
                        questions_per_type=questions_per_type,
                        overwrite_cache=overwrite_cache,
                    )
                if max_questions and max_questions > 0:
                    questions = questions[:max_questions]

                answers_map, _ = await self.answerer.answer_questions(
                    target_dataset=tgt_ds,
                    questions=questions,
                    algo_name=target_algo_name,
                    workload_title=wl_title,
                    workload_description=wl_desc,
                    batch_size=batch_size,
                    parallel_num=parallel_num,
                )
                scored = await asyncio.gather(
                    *[self.scorer.score_question(q, answers_map.get(str(q.get("id", "")), {})) for q in questions]
                )
                for item in scored:
                    item["round"] = r_idx
                    item["workload_key"] = wl_key
                    round_scored.append(item)
                    all_scored.append(item)

            r_stat = _summarize_group(round_scored)
            rounds_summary.append({
                "round": r_idx,
                "questions_count": r_stat["count"],
                "average_score": r_stat["average_score"],
                "accuracy_threshold_4": r_stat["accuracy_t4"],
                "accuracy_threshold_5": r_stat["accuracy_t5"],
                "mean_absolute_error": r_stat["mae"],
                "scored_questions": round_scored,
            })

        overall = _summarize_group(all_scored)
        mean_t4, std_t4 = _mean_std([r["accuracy_threshold_4"] for r in rounds_summary])
        mean_sc, std_sc = _mean_std([r["average_score"] for r in rounds_summary])

        by_type, by_attr, by_matrix = {}, {}, {}
        for q in all_scored:
            by_type.setdefault(q["question_type"], []).append(q)
            by_attr.setdefault(q["attr_type"], []).append(q)
            by_matrix.setdefault(f"{q['attr_type']}__{q['question_type']}", []).append(q)

        report = {
            "target_algo": target_algo_name,
            "gt_algo": gt_algo_name,
            "model_name": self.model_name,
            "workload_file": workload_file_path,
            "num_rounds": num_rounds,
            "total_questions_evaluated": overall["count"],
            "primary_accuracy_t4": overall["accuracy_t4"],
            "overall_average_score": overall["average_score"],
            "overall_mae": overall["mae"],
            "multi_round_stats": {
                "accuracy_t4_mean": round(mean_t4, 4),
                "accuracy_t4_std": round(std_t4, 4),
                "average_score_mean": round(mean_sc, 4),
                "average_score_std": round(std_sc, 4),
            },
            "breakdown_by_question_type": {k: _summarize_group(v) for k, v in by_type.items()},
            "breakdown_by_attr_type": {k: _summarize_group(v) for k, v in by_attr.items()},
            "breakdown_by_matrix_cell": {k: _summarize_group(v) for k, v in by_matrix.items()},
            "duration_seconds": round(time.time() - t0, 2),
            "llm_usage": self.agent.usage,
            "rounds_summary": rounds_summary,
        }
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report
