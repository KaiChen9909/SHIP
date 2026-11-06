"""Deterministic numerical/binary rubrics + LLM comparison judge for Closed-Ended QA."""

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

_COMPARE_JUDGE_PROMPT = os.path.join(os.path.dirname(__file__), "prompts", "score_compare_prompt.json")


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


class QAScorer:
    """Scores candidate answers on a 1–5 rubric (primary metric: Accuracy@T4, i.e. score >= 4)."""

    def __init__(self, agent_executor: CodeAgentExecutor | None = None, max_judge_concurrency: int = 10):
        self.agent = agent_executor
        self._judge_sem = asyncio.Semaphore(max(1, max_judge_concurrency))

    def score_numerical(self, candidate_val: Any, gt_val: Any) -> tuple[int, float | None, float | None, str]:
        """Score numerical proportion questions using absolute and relative error thresholds."""
        c_f, g_f = parse_float_val(candidate_val), parse_float_val(gt_val)
        if c_f is None or g_f is None:
            return 1, None, None, f"Unparseable numeric answer (candidate={candidate_val}, gt={gt_val})."
        c_f, g_f = normalize_percentage_pair(c_f, g_f)
        abs_err = abs(c_f - g_f)
        rel_err = abs_err / max(abs(g_f), 1e-4)
        if abs_err <= 0.05 or rel_err <= 0.10:
            score = 5
        elif abs_err <= 0.10 or rel_err <= 0.20:
            score = 4
        elif abs_err <= 0.25 or rel_err <= 0.50:
            score = 3
        elif abs_err <= 0.50:
            score = 2
        else:
            score = 1
        return score, abs_err, rel_err, (
            f"Candidate={c_f:.4f}, GT={g_f:.4f} | AbsErr={abs_err:.4f}, RelErr={rel_err:.4f}"
        )

    def score_binary(self, candidate_val: Any, gt_val: Any) -> tuple[int, str]:
        """Score Yes/No questions deterministically (5 if matched, 1 otherwise; 'I don't know' -> 1)."""
        c_s, g_s = str(candidate_val or "").strip().lower(), str(gt_val or "").strip().lower()
        c_yes, g_yes = bool(re.search(r"\b(yes|true)\b", c_s)), bool(re.search(r"\b(yes|true)\b", g_s))
        c_no, g_no = bool(re.search(r"\b(no|false)\b", c_s)), bool(re.search(r"\b(no|false)\b", g_s))
        if (c_yes and g_yes and not c_no) or (c_no and g_no and not c_yes):
            return 5, f"Matched binary verdict ({candidate_val} == {gt_val})."
        return 1, f"Mismatched binary verdict (candidate={candidate_val}, gt={gt_val})."

    async def score_comparison(
        self, candidate_val: Any, gt_val: Any, question_text: str, gt_expl: str, cand_expl: str
    ) -> tuple[int, str]:
        """Score comparison questions: exact match, else LLM judge; word overlap only if the judge fails."""
        c_s, g_s = str(candidate_val or "").strip().lower(), str(gt_val or "").strip().lower()
        if not g_s:
            return 1, "Empty ground-truth answer; question cannot be scored."
        if not c_s or c_s in ("none", "n/a", "null", "i don't know", "error"):
            return 1, f"Abstained or empty candidate answer (GT={gt_val})."
        if c_s == g_s:
            return 5, f"Exact concept match ('{candidate_val}')."
        # Substring containment is NOT treated as a match ("B rather than A" contains "A").

        if self.agent is not None:
            try:
                async with self._judge_sem:
                    tmpl = DatasetTemplateManager.load_prompt_template(_COMPARE_JUDGE_PROMPT)
                    prompt = safe_format_prompt(
                        tmpl,
                        question=question_text,
                        question_type="compare",
                        gt_answer=str(gt_val),
                        gt_explanation=gt_expl,
                        candidate_answer=str(candidate_val),
                        candidate_explanation=cand_expl,
                    )
                    raw = await self.agent.llm_call(prompt, temperature=0.0)
                parsed = extract_json_dict(raw)
                if parsed and "score" in parsed:
                    sc = max(1, min(5, int(parsed["score"])))
                    return sc, f"LLM Judge: {parsed.get('reason', '')}"
            except Exception as e:  # noqa: BLE001
                logger.warning("Comparison LLM judge fallback triggered: %s", e)

        # Fallback when the judge is unavailable: word overlap cannot tell which side won, so it
        # is capped at 3 ("inconclusive") and never counts as correct at threshold 4.
        stop = {"the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "on", "at", "is", "are"}
        c_words = set(re.findall(r"\w+", c_s)) - stop
        g_words = set(re.findall(r"\w+", g_s)) - stop
        if not g_words:
            return 1, "Ground-truth answer has no content words."
        ratio = len(c_words & g_words) / len(g_words)
        if ratio > 0.0:
            return 3, f"Judge unavailable; word overlap {ratio:.0%} (direction unverified)."
        return 1, f"Judge unavailable; no concept overlap ('{candidate_val}' vs '{gt_val}')."

    async def score_question(self, question_dict: dict[str, Any], answer_dict: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a question + candidate answer to the matching scorer."""
        q_type = str(question_dict.get("question_type", "yes_no")).lower()
        gt_val = question_dict.get("gt_answer", question_dict.get("gt_target_value"))
        cand_val = answer_dict.get("answer_value")
        abs_err = rel_err = None

        if any(k in q_type for k in ("ask_for_number", "number", "percentage", "proportion", "numerical")):
            score, abs_err, rel_err, reason = self.score_numerical(cand_val, gt_val)
        elif any(k in q_type for k in ("yes_no", "binary", "boolean")):
            score, reason = self.score_binary(cand_val, gt_val)
        else:
            score, reason = await self.score_comparison(
                cand_val,
                gt_val,
                str(question_dict.get("question", "")),
                str(question_dict.get("gt_explanation", "")),
                str(answer_dict.get("explanation", "")),
            )

        return {
            "id": question_dict.get("id", ""),
            "question": question_dict.get("question", ""),
            "attr_type": question_dict.get("attr_type", "single_attr"),
            "question_type": q_type,
            "level_info": question_dict.get("level_info", ""),
            "gt_answer": gt_val,
            "candidate_answer": cand_val,
            "candidate_display_answer": answer_dict.get("display_answer", ""),
            "candidate_explanation": answer_dict.get("explanation", ""),
            "score": score,
            "supported_t4": score >= 4,
            "absolute_error": abs_err,
            "relative_error": rel_err,
            "reason": reason,
        }
