"""Stratified Ground-Truth question bank generator (single_attr/multi_attr x yes_no/compare/ask_for_number)."""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import time
from typing import Any, Sequence

from eval.common.code_agent import CodeAgentExecutor, extract_json_dict
from eval.common.dataset_manager import DatasetContext, DatasetTemplateManager, safe_format_prompt

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def get_attr_instruction(attr_type: str) -> str:
    """Return attribute scope instruction for single_attr vs multi_attr question generation."""
    if "single" in attr_type.lower():
        return (
            "ATTRIBUTE REQUIREMENT & HIERARCHICAL STRATIFICATION (single_attr):\n"
            "- Formulate questions that query ONLY A SINGLE attribute/dimension (within one taxonomy dimension).\n"
            "- DO NOT combine, correlate, or condition on other orthogonal attributes.\n"
            "- BALANCED DEPTH COVERAGE: Balance questions across Level-1, Level-2, and Level-3.\n"
            "- DIVERSITY & NON-REPETITION: Explore distinct categories and subtrees without repeating prior questions.\n"
            "- STATISTICAL SIGNIFICANCE CONSTRAINT: Queried slices must represent >=0.5%–1% of the dataset."
        )
    return (
        "ATTRIBUTE REQUIREMENT & HIERARCHICAL STRATIFICATION (multi_attr):\n"
        "- Formulate questions querying the relationship, conditional distribution, or joint distribution across "
        "AT LEAST TWO DIFFERENT orthogonal attributes/dimensions.\n"
        "- NEVER ask questions depending on only a single attribute.\n"
        "- GRANULAR EXPLORATION & STATISTICAL SIGNIFICANCE: Explore cross-tabulations across levels while ensuring "
        "each target slice represents >=0.5%–1% of the dataset."
    )


def get_proposing_prompt_file(question_type: str) -> str:
    """Map question_type to its prompt filename in eval/qa/prompts/."""
    q = question_type.lower()
    if any(k in q for k in ("compare", "comparison", "contrast")):
        return "propose_compare_prompt.json"
    if any(k in q for k in ("ask_for_number", "number", "numerical", "percentage", "proportion")):
        return "propose_ask_for_number_prompt.json"
    return "propose_yes_no_prompt.json"


class QuestionGenerator:
    """Generates and caches grounded benchmark questions on the Ground-Truth dataset."""

    def __init__(self, agent_executor: CodeAgentExecutor, cache_dir: str = ".cache/eval_questions"):
        self.agent = agent_executor
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    async def generate_typed_questions(
        self,
        dataset: DatasetContext,
        attr_type: str,
        question_type: str,
        k: int = 50,
        batch_size: int = 10,
        workload_content: str = "",
        workload_title: str = "",
        workload_description: str = "",
        workload_key: str = "",
    ) -> list[dict[str, Any]]:
        """Generate k deduplicated questions for one (attr_type, question_type) cell."""
        template_str = DatasetTemplateManager.load_prompt_template(
            os.path.join(_PROMPTS_DIR, get_proposing_prompt_file(question_type))
        )
        attr_instruction = get_attr_instruction(attr_type)
        collected: list[dict[str, Any]] = []
        retries = 0

        while len(collected) < k and retries < 6:
            needed = min(batch_size, k - len(collected))
            if collected:
                q_lines = "\n".join(
                    f"- {q['question']} ({q.get('level_info') or 'unspecified'} question)" for q in collected
                )
                lvl_counts = collections.Counter(q.get("level_info") for q in collected if q.get("level_info"))
                max_c = max(lvl_counts.values(), default=0)
                under_rep = [lv for lv in ("level_1", "level_2", "level_3") if lvl_counts.get(lv, 0) < max(1, max_c // 2)]
                lvl_hint = (
                    f"\n- DIVERSITY RE-BALANCING: Prioritize under-represented levels: {', '.join(under_rep)}."
                    if under_rep else ""
                )
                existing_instruction = (
                    f"PREVIOUSLY GENERATED QUESTIONS IN THIS BANK ({len(collected)} created; levels={dict(lvl_counts)}):\n"
                    f"{q_lines}\n\n"
                    "STRICT NON-DUPLICATION & DIVERSITY RULE:\n"
                    "- Do NOT repeat or paraphrase any question, slice, or comparison pair above.\n"
                    f"- Target unexplored categories and subtrees across level_1, level_2, level_3.{lvl_hint}"
                )
            else:
                existing_instruction = ""

            prompt = safe_format_prompt(
                template_str,
                dataset_description=dataset.intro_text or dataset.algo_name,
                workload_file=workload_content,
                workload_title=workload_title,
                workload_description=workload_description,
                attr_type_instruction=attr_instruction,
                existing_questions_instruction=existing_instruction,
                attr_type=attr_type,
                question_type=question_type,
                data_dir=dataset.data_dir,
                data_file_path=dataset.main_csv_path,
                category_tree_path=dataset.category_tree_path,
                dataset_info_path=dataset.dataset_info_path,
                algo_name=dataset.algo_name,
                data_files=", ".join(dataset.data_files),
                intro_text=dataset.intro_text,
                workload_content=workload_content,
                k=needed,
            )

            parsed: Any = None
            for attempt in range(3):
                try:
                    raw, _ = await self.agent.run_agent_query(
                        dataset, prompt, algo_name=dataset.algo_name, temperature=0.9, task_type="question_generation"
                    )
                    parsed = extract_json_dict(raw)
                    if parsed:
                        break
                except Exception as e:  # noqa: BLE001
                    logger.warning("Question batch (%s, %s) attempt %d failed: %s", attr_type, question_type, attempt + 1, e)

            if isinstance(parsed, dict):
                for wrap_k in ("questions", "output", "data", "results"):
                    if wrap_k in parsed and isinstance(parsed[wrap_k], (dict, list)):
                        parsed = parsed[wrap_k]
                        break

            items = list(parsed.values()) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
            seen = {q["question"].strip().lower() for q in collected}
            added = 0
            for q_val in items:
                if not isinstance(q_val, dict):
                    continue
                q_text = str(q_val.get("question", "")).strip()
                if not q_text or q_text.lower() in seen:
                    continue
                seen.add(q_text.lower())
                gt_ans = q_val.get("gt_answer", q_val.get("gt_target_value", q_val.get("answer", "")))
                collected.append({
                    "id": f"{attr_type}_{question_type}_{len(collected) + 1}",
                    "question": q_text,
                    "attr_type": attr_type,
                    "question_type": question_type,
                    "level_info": str(q_val.get("level_info", q_val.get("level", ""))).strip(),
                    "gt_answer": gt_ans,
                    "gt_target_value": gt_ans,
                    "gt_display_value": str(q_val.get("gt_display_value", gt_ans)),
                    "gt_explanation": str(q_val.get("gt_explanation", q_val.get("explanation", ""))),
                    "verification_code": str(q_val.get("verification_code", "")),
                    "workload_key": workload_key,
                    "workload_title": workload_title,
                })
                added += 1
                if len(collected) >= k:
                    break

            if added == 0:
                retries += 1
                await asyncio.sleep(2.0 * retries)
            else:
                retries = 0
                logger.info("Collected %d/%d questions for (%s, %s)", len(collected), k, attr_type, question_type)

        return collected[:k]

    async def generate_question_bank(
        self,
        dataset: DatasetContext,
        workload_content: str = "",
        workload_name: str = "default",
        workload_title: str = "",
        workload_description: str = "",
        workload_key: str = "",
        attr_types: Sequence[str] = ("single_attr", "multi_attr"),
        question_types: Sequence[str] = ("yes_no", "compare", "ask_for_number"),
        questions_per_type: int = 50,
        overwrite_cache: bool = False,
        output_path: str | None = None,
        max_concurrency: int = 4,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate or load the complete stratified question bank across attr_types x question_types."""
        bank_path = output_path or os.path.join(dataset.data_dir, f"{workload_name}_question_bank.json")
        fallback_path = os.path.join(self.cache_dir, f"{dataset.get_id()}_{workload_name}_question_bank.json")

        if not overwrite_cache:
            for cand in (bank_path, fallback_path):
                if os.path.isfile(cand):
                    with open(cand, encoding="utf-8") as f:
                        qs = json.load(f)
                    q_set, a_set = set(question_types), set(attr_types)
                    filtered = [q for q in qs if q.get("question_type") in q_set and q.get("attr_type") in a_set]
                    if filtered:
                        logger.info("Loaded %d cached GT questions from %s", len(filtered), cand)
                        return cand, filtered

        sem = asyncio.Semaphore(max(1, max_concurrency))

        async def _cell(a_t: str, q_t: str) -> list[dict[str, Any]]:
            async with sem:
                return await self.generate_typed_questions(
                    dataset=dataset,
                    attr_type=a_t,
                    question_type=q_t,
                    k=questions_per_type,
                    workload_content=workload_content,
                    workload_title=workload_title,
                    workload_description=workload_description,
                    workload_key=workload_key,
                )

        cells = await asyncio.gather(*[_cell(a_t, q_t) for a_t in attr_types for q_t in question_types])
        all_questions = [q for cell in cells for q in cell]

        save_target = bank_path
        try:
            with open(save_target, "w", encoding="utf-8") as f:
                json.dump(all_questions, f, indent=2, ensure_ascii=False)
        except OSError:
            save_target = fallback_path
            with open(save_target, "w", encoding="utf-8") as f:
                json.dump(all_questions, f, indent=2, ensure_ascii=False)
        return save_target, all_questions
