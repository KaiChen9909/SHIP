"""Symmetric 2-pass Open-Ended QA evaluator (Recall@T4, Precision@T4, F1@T4 across R rounds)."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any

from eval.common.code_agent import DEFAULT_EVAL_MODEL, CodeAgentExecutor
from eval.common.dataset_manager import DatasetTemplateManager

from .statement_extractor import StatementExtractor
from .statement_verifier import StatementVerifier

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    return m, math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def _compute_threshold_metrics(p1_scores: list[float], p2_scores: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in (3, 4, 5):
        p1_sup = sum(1 for s in p1_scores if s >= t)
        p2_sup = sum(1 for s in p2_scores if s >= t)
        rec = p1_sup / len(p1_scores) if p1_scores else 0.0
        prec = p2_sup / len(p2_scores) if p2_scores else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        out[f"threshold_{t}"] = {
            "min_supported_score": t,
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "pass1_gt_supported_count": p1_sup,
            "pass2_target_supported_count": p2_sup,
        }
    return out


class SymmetricEvaluator:
    """Orchestrates 5-round bi-directional GT <-> Target SHIP open-ended evaluation."""

    def __init__(
        self,
        model_name: str = DEFAULT_EVAL_MODEL,
        vertexai: bool = False,
        cache_dir: str = ".cache/eval_statements",
    ):
        self.model_name = model_name
        self.agent = CodeAgentExecutor(model_name=model_name, vertexai=vertexai)
        self.extractor = StatementExtractor(self.agent, cache_dir=cache_dir)
        self.verifier = StatementVerifier(self.agent)

    async def evaluate(
        self,
        target_data_dir: str,
        workload_file_path: str,
        gt_data_dir: str,
        target_algo_name: str = "ship",
        gt_algo_name: str = "groundtruth",
        dataset_name: str = "wildchat",
        dataset_description: str = "",
        num_rounds: int = 5,
        k: int | None = None,
        validation_batch_size: int = 10,
        parallel_num: int = 5,
        output_report_path: str | None = None,
        overwrite_cache: bool = False,
        overwrite_gt_cache: bool = False,
    ) -> dict[str, Any]:
        """Run symmetric 2-pass open-ended QA evaluation across `num_rounds`."""
        t0 = time.time()
        gt_ext = os.path.join(_PROMPTS_DIR, "gt_answer_prompt.json")
        gt_val = os.path.join(_PROMPTS_DIR, "gt_score_prompt.json")
        tgt_ext = os.path.join(_PROMPTS_DIR, "table_answer_prompt.json")
        tgt_val = os.path.join(_PROMPTS_DIR, "table_score_prompt.json")

        gt_ds = DatasetTemplateManager.load_dataset(gt_algo_name, gt_data_dir, dataset_name, dataset_description)
        tgt_ds = DatasetTemplateManager.load_dataset(target_algo_name, target_data_dir, dataset_name, dataset_description)
        workload_dict = DatasetTemplateManager.load_workload_dict(workload_file_path)
        wl_id = os.path.splitext(os.path.basename(workload_file_path))[0]
        report_path = output_report_path or os.path.join(tgt_ds.data_dir, f"eval_report_{wl_id}.json")

        cached_wl: dict[str, Any] = {}
        if not overwrite_cache and os.path.isfile(report_path):
            try:
                with open(report_path, encoding="utf-8") as f:
                    cached_wl = json.load(f).get("workload_queries_detail", {})
            except Exception:  # noqa: BLE001
                cached_wl = {}

        rounds_summary: list[dict[str, Any]] = []
        workload_results: dict[str, Any] = {}
        all_p1_scores: list[float] = []
        all_p2_scores: list[float] = []

        for r_idx in range(1, num_rounds + 1):
            logger.info("=== Open-Ended Symmetric QA Round %d / %d ===", r_idx, num_rounds)
            r_p1_vers: list[dict[str, Any]] = []
            r_p2_vers: list[dict[str, Any]] = []

            for wl_key, wl_item in workload_dict.items():
                wl_title, wl_desc = wl_item.get("title", ""), wl_item.get("description", "")
                wl_name = f"{wl_id}_wl{wl_key}_r{r_idx}" if num_rounds > 1 else f"{wl_id}_wl{wl_key}"

                prev_entry = (
                    cached_wl.get(str(wl_key), {}).get("rounds", {}).get(str(r_idx))
                    or cached_wl.get(str(wl_key), {}).get("rounds", {}).get(r_idx)
                )
                if (
                    not overwrite_cache
                    and prev_entry
                    and prev_entry.get("pass1_gt_to_target_verifications")
                    and prev_entry.get("pass2_target_to_gt_verifications")
                ):
                    p1_vers = prev_entry["pass1_gt_to_target_verifications"]
                    p2_vers = prev_entry["pass2_target_to_gt_verifications"]
                    gt_stmts_path = prev_entry.get("gt_answer_path", "")
                    tgt_stmts_path = prev_entry.get("target_answer_path", "")
                else:
                    # Pass 1: Extract on GT -> Verify on Target SHIP (Recall).
                    gt_stmts_path, _ = await self.extractor.extract_or_load(
                        dataset=gt_ds,
                        workload_content=json.dumps(wl_item),
                        workload_name=wl_name,
                        prompt_template_path=gt_ext,
                        k=k,
                        overwrite_cache=overwrite_gt_cache,
                        algo_name=gt_algo_name,
                        workload_title=wl_title,
                        workload_description=wl_desc,
                        workload_key=str(wl_key),
                    )
                    p1_vers, _ = await self.verifier.verify(
                        recipient_dataset=tgt_ds,
                        statements_path=gt_stmts_path,
                        workload_content=json.dumps(wl_item),
                        prompt_template_path=tgt_val,
                        batch_size=validation_batch_size,
                        parallel_num=parallel_num,
                        algo_name=target_algo_name,
                        workload_title=wl_title,
                        workload_description=wl_desc,
                        workload_key=str(wl_key),
                    )

                    # Pass 2: Extract on Target SHIP -> Verify on GT (Precision).
                    tgt_stmts_path, _ = await self.extractor.extract_or_load(
                        dataset=tgt_ds,
                        workload_content=json.dumps(wl_item),
                        workload_name=wl_name,
                        prompt_template_path=tgt_ext,
                        k=k,
                        overwrite_cache=overwrite_cache,
                        algo_name=target_algo_name,
                        workload_title=wl_title,
                        workload_description=wl_desc,
                        workload_key=str(wl_key),
                    )
                    p2_vers, _ = await self.verifier.verify(
                        recipient_dataset=gt_ds,
                        statements_path=tgt_stmts_path,
                        workload_content=json.dumps(wl_item),
                        prompt_template_path=gt_val,
                        batch_size=validation_batch_size,
                        parallel_num=parallel_num,
                        algo_name=gt_algo_name,
                        workload_title=wl_title,
                        workload_description=wl_desc,
                        workload_key=str(wl_key),
                    )

                r_p1_vers.extend(p1_vers)
                r_p2_vers.extend(p2_vers)
                workload_results.setdefault(
                    str(wl_key),
                    {"workload_key": str(wl_key), "title": wl_title, "description": wl_desc, "rounds": {}},
                )["rounds"][str(r_idx)] = {
                    "round": r_idx,
                    "gt_answer_path": gt_stmts_path,
                    "target_answer_path": tgt_stmts_path,
                    "pass1_gt_to_target_verifications": p1_vers,
                    "pass2_target_to_gt_verifications": p2_vers,
                }

            p1_sc = [float(v.get("score") or 0.0) for v in r_p1_vers]
            p2_sc = [float(v.get("score") or 0.0) for v in r_p2_vers]
            all_p1_scores.extend(p1_sc)
            all_p2_scores.extend(p2_sc)
            t_metrics = _compute_threshold_metrics(p1_sc, p2_sc)

            rounds_summary.append({
                "round": r_idx,
                "pass1_statements_count": len(p1_sc),
                "pass2_statements_count": len(p2_sc),
                "primary_f1_quality_score_t4": t_metrics["threshold_4"]["f1_score"],
                "primary_recall_t4": t_metrics["threshold_4"]["recall"],
                "primary_precision_t4": t_metrics["threshold_4"]["precision"],
                "average_validation_scores": {
                    "pass1_gt_on_target_avg": round(sum(p1_sc) / len(p1_sc), 4) if p1_sc else 0.0,
                    "pass2_target_on_gt_avg": round(sum(p2_sc) / len(p2_sc), 4) if p2_sc else 0.0,
                },
                "metrics_by_threshold": t_metrics,
            })

            # Save incremental checkpoint after each round.
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "gt_algo": gt_algo_name,
                        "target_algo": target_algo_name,
                        "model_name": self.model_name,
                        "workload_file": workload_file_path,
                        "rounds_completed": r_idx,
                        "num_rounds_target": num_rounds,
                        "rounds_summary": rounds_summary,
                        "workload_queries_detail": workload_results,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        pooled = _compute_threshold_metrics(all_p1_scores, all_p2_scores)
        f1_m, f1_s = _mean_std([r["primary_f1_quality_score_t4"] for r in rounds_summary])
        rec_m, rec_s = _mean_std([r["primary_recall_t4"] for r in rounds_summary])
        prec_m, prec_s = _mean_std([r["primary_precision_t4"] for r in rounds_summary])

        final_report = {
            "gt_algo": gt_algo_name,
            "target_algo": target_algo_name,
            "model_name": self.model_name,
            "workload_file": workload_file_path,
            "num_rounds": num_rounds,
            "primary_f1_quality_score_t4": round(f1_m, 4),
            "primary_recall_t4": round(rec_m, 4),
            "primary_precision_t4": round(prec_m, 4),
            "multi_round_stats": {
                "f1_t4_mean": round(f1_m, 4),
                "f1_t4_std": round(f1_s, 4),
                "recall_t4_mean": round(rec_m, 4),
                "recall_t4_std": round(rec_s, 4),
                "precision_t4_mean": round(prec_m, 4),
                "precision_t4_std": round(prec_s, 4),
            },
            "pooled_metrics_by_threshold": pooled,
            "duration_seconds": round(time.time() - t0, 2),
            "llm_usage": self.agent.usage,
            "rounds_summary": rounds_summary,
            "workload_queries_detail": workload_results,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        return final_report
