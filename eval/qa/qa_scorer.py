"""LLM-judge scorer for Closed-Ended QA: every question type is scored on the paper's 1-5 rubric
(Yes/No rule, comparison rubric, numerical rubric) by an LLM judge with structured output."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from typing import Any

from eval.common.code_agent import CodeAgentExecutor, extract_json_dict
from eval.common.dataset_manager import DatasetTemplateManager, safe_format_prompt

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_JUDGE_PROMPTS = {
    "yes_no": os.path.join(_PROMPTS_DIR, "score_yes_no_prompt.json"),
    "compare": os.path.join(_PROMPTS_DIR, "score_compare_prompt.json"),
    "ask_for_number": os.path.join(_PROMPTS_DIR, "score_ask_for_number_prompt.json"),
}

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
}


def canonical_question_type(q_type: str) -> str:
    q = q_type.lower()
    if any(k in q for k in ("ask_for_number", "number", "percentage", "proportion", "numerical")):
        return "ask_for_number"
    if any(k in q for k in ("compare", "comparison", "contrast")):
        return "compare"
    return "yes_no"


def parse_float_val(val: Any) -> float | None:
    """Parse a float, a proportion, or a percentage string ("39.94%", "0.5 %", "0.3994 (39.94%)")."""
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return None if (math.isnan(val) or math.isinf(val)) else float(val)
    if isinstance(val, str):
        clean = val.strip().replace(",", "")
        m = re.search(r"([-+]?\d*\.?\d+)\s*(%|percent\b)?", clean, re.IGNORECASE)
        if m:
            f = float(m.group(1))
            if m.group(2):  # the first number itself is a percentage
                f /= 100.0
            return f
    return None


def normalize_percentage_pair(c: float, g: float) -> tuple[float, float]:
    """Align candidate and GT proportions onto a common [0, 1] scale."""
    if 1.0 < g <= 100.0:
        g /= 100.0
    if 1.0 < c <= 100.0 and g <= 1.0:
        c /= 100.0
    elif c <= 1.0 and g > 1.0:
        c *= 100.0
    return c, g


def numeric_errors(candidate_val: Any, gt_val: Any) -> tuple[float | None, float | None]:
    """Absolute / relative error of numerical answers. Reported as auxiliary MAE only; never used for scoring."""
    c_f, g_f = parse_float_val(candidate_val), parse_float_val(gt_val)
    if c_f is None or g_f is None:
        return None, None
    c_f, g_f = normalize_percentage_pair(c_f, g_f)
    abs_err = abs(c_f - g_f)
    return abs_err, abs_err / max(abs(g_f), 1e-4)


class QAScorer:
    """Scores candidate answers on a 1-5 rubric (primary metric: Accuracy@T4, i.e. score >= 4)."""

    def __init__(self, agent_executor: CodeAgentExecutor, max_judge_concurrency: int = 10, parse_retries: int = 2):
        self.agent = agent_executor
        self.parse_retries = parse_retries
        self._judge_sem = asyncio.Semaphore(max(1, max_judge_concurrency))

    async def judge(
        self, q_type: str, question: str, gt_val: Any, gt_expl: str, cand_val: Any, cand_expl: str
    ) -> tuple[int | None, str]:
        """Return (score, reason); score is None if the judge failed after all retries."""
        prompt = safe_format_prompt(
            DatasetTemplateManager.load_prompt_template(_JUDGE_PROMPTS[q_type]),
            question=question,
            question_type=q_type,
            gt_answer=str(gt_val),
            gt_explanation=gt_expl,
            candidate_answer="(no answer)" if cand_val is None else str(cand_val),
            candidate_explanation=cand_expl,
        )
        err: Exception | str = "unparseable judge output"
        for k in range(self.parse_retries + 1):
            try:
                async with self._judge_sem:
                    raw = await self.agent.llm_call(prompt, temperature=0.0, schema=SCORE_SCHEMA,
                                                    cache_salt=f"try{k}" if k else "")
                parsed = extract_json_dict(raw)
                if parsed and "score" in parsed:
                    return max(1, min(5, int(parsed["score"]))), str(parsed.get("reason", ""))
            except Exception as e:  # noqa: BLE001
                err = e
        logger.warning("QA judge failed (%s): %s", q_type, err)
        return None, f"Judge failed: {err}"

    async def score_question(self, question_dict: dict[str, Any], answer_dict: dict[str, Any]) -> dict[str, Any]:
        raw_type = str(question_dict.get("question_type", "yes_no")).lower()
        q_type = canonical_question_type(raw_type)
        gt_val = question_dict.get("gt_answer", question_dict.get("gt_target_value"))
        cand_val = answer_dict.get("answer_value")

        score, reason = await self.judge(
            q_type,
            str(question_dict.get("question", "")),
            gt_val,
            str(question_dict.get("gt_explanation", "")),
            cand_val,
            str(answer_dict.get("explanation", "")),
        )
        judge_failed = score is None
        if judge_failed:
            score = 1  # counted as unsupported; flagged so failures are visible in the report

        abs_err = rel_err = None
        if q_type == "ask_for_number":
            abs_err, rel_err = numeric_errors(cand_val, gt_val)

        return {
            "id": question_dict.get("id", ""),
            "question": question_dict.get("question", ""),
            "attr_type": question_dict.get("attr_type", "single_attr"),
            "question_type": raw_type,
            "level_info": question_dict.get("level_info", ""),
            "gt_answer": gt_val,
            "candidate_answer": cand_val,
            "candidate_display_answer": answer_dict.get("display_answer", ""),
            "candidate_explanation": answer_dict.get("explanation", ""),
            "score": score,
            "supported_t4": score >= 4,
            "judge_failed": judge_failed,
            "absolute_error": abs_err,
            "relative_error": rel_err,
            "reason": reason,
        }
