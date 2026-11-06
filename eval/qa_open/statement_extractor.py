"""2-stage open-ended report generation and atomic statement decomposition."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from eval.common.code_agent import CodeAgentExecutor, extract_json_dict
from eval.common.dataset_manager import DatasetContext, DatasetTemplateManager, safe_format_prompt

logger = logging.getLogger(__name__)

_SPLIT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "statement_split_prompt.json")

_META_ARTIFACTS = (
    "report has been generated",
    "report is saved at",
    "saved at /",
    "markdown analysis report is saved",
    "file:///",
    ".md.",
)


class StatementExtractor:
    """Generates aggregated Markdown reports via REPL code execution and splits them into atomic claims."""

    def __init__(self, agent_executor: CodeAgentExecutor, cache_dir: str = ".cache/eval_statements"):
        self.agent = agent_executor
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    async def extract_or_load(
        self,
        dataset: DatasetContext,
        workload_content: str,
        workload_name: str,
        prompt_template_path: str,
        k: int | None = None,
        overwrite_cache: bool = False,
        algo_name: str | None = None,
        workload_title: str = "",
        workload_description: str = "",
        workload_key: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return (statements_json_path, statements_list) via 2-stage report -> split pipeline."""
        stmt_path = os.path.join(dataset.data_dir, f"{workload_name}_statements.json")
        fallback_stmt = os.path.join(self.cache_dir, f"{dataset.get_id()}_{workload_name}_statements.json")
        doc_path = os.path.join(dataset.data_dir, f"{workload_name}_analysis_doc.md")
        fallback_doc = os.path.join(self.cache_dir, f"{dataset.get_id()}_{workload_name}_analysis_doc.md")

        if not overwrite_cache:
            for cand in (stmt_path, fallback_stmt):
                if os.path.isfile(cand):
                    with open(cand, encoding="utf-8") as f:
                        stmts = json.load(f)
                    if stmts:
                        logger.info("Loaded %d cached statements from %s", len(stmts), cand)
                        return cand, stmts

        # Stage 1: Generate or load analytical Markdown report.
        analysis_doc = ""
        if not overwrite_cache:
            for cand in (doc_path, fallback_doc):
                if os.path.isfile(cand):
                    with open(cand, encoding="utf-8") as f:
                        analysis_doc = f.read().strip()
                    if analysis_doc:
                        break

        if not analysis_doc:
            tmpl = DatasetTemplateManager.load_prompt_template(prompt_template_path)
            doc_prompt = safe_format_prompt(
                tmpl,
                dataset_description=dataset.intro_text or dataset.algo_name,
                workload_file=workload_content,
                workload_title=workload_title,
                workload_description=workload_description,
                data_dir=dataset.data_dir,
                data_file_path=dataset.full_path_csv_path,
                category_tree_path=dataset.category_tree_path,
                dataset_info_path=dataset.dataset_info_path,
                algo_name=dataset.algo_name,
                data_files=", ".join(dataset.data_files),
                intro_text=dataset.intro_text,
                workload_content=workload_content,
                k=k or "exhaustive",
                num_statements=k or "exhaustive",
            )
            raw_doc, _ = await self.agent.run_agent_query(
                dataset,
                doc_prompt,
                algo_name=algo_name or dataset.algo_name,
                temperature=0.0,
                task_type="aggregated_analysis",
            )
            analysis_doc = raw_doc.strip()
            for target_doc in (doc_path, fallback_doc):
                try:
                    with open(target_doc, "w", encoding="utf-8") as f:
                        f.write(analysis_doc)
                    break
                except OSError:
                    continue

        # Stage 2: Decompose Markdown report into standalone atomic statements.
        split_tmpl = DatasetTemplateManager.load_prompt_template(_SPLIT_PROMPT_PATH)
        split_prompt = safe_format_prompt(
            split_tmpl,
            workload_title=workload_title or workload_name,
            workload_description=workload_description or workload_content,
            analysis_document=analysis_doc,
            k=k if k and k > 0 else "exhaustive",
            num_statements=k if k and k > 0 else "exhaustive",
        )
        split_raw = await self.agent.llm_call(split_prompt, temperature=0.0)
        parsed: Any = extract_json_dict(split_raw)
        fence = "`" * 3
        if parsed is None:
            m = re.search(rf"{fence}(?:json)?\s*(\[.*?\])\s*{fence}", split_raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:  # noqa: BLE001
                    parsed = {}

        if isinstance(parsed, dict):
            for wrap_k in ("statements", "output", "data", "results"):
                if wrap_k in parsed and isinstance(parsed[wrap_k], (list, dict)):
                    parsed = parsed[wrap_k]
                    break

        raw_texts: list[str] = []
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, str):
                    raw_texts.append(v)
                elif isinstance(v, dict) and "statement" in v:
                    raw_texts.append(str(v["statement"]))
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    raw_texts.append(item)
                elif isinstance(item, dict) and "statement" in item:
                    raw_texts.append(str(item["statement"]))

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for txt in raw_texts:
            clean = txt.strip()
            low = clean.lower()
            if not clean or low in seen or any(p in low for p in _META_ARTIFACTS):
                continue
            seen.add(low)
            deduped.append({
                "id": f"statement {len(deduped) + 1}",
                "statement": clean,
                "workload_key": workload_key,
                "workload_title": workload_title,
                "workload_description": workload_description,
            })
            if k and k > 0 and len(deduped) >= k:
                break

        if not deduped and analysis_doc:
            for para in analysis_doc.split("\n\n"):
                p_clean = para.strip().lstrip("#-* ").strip()
                if len(p_clean) > 20 and not p_clean.startswith(fence):
                    deduped.append({"id": f"statement {len(deduped) + 1}", "statement": p_clean})
                    if k and k > 0 and len(deduped) >= k:
                        break

        save_target = stmt_path
        try:
            with open(save_target, "w", encoding="utf-8") as f:
                json.dump(deduped, f, indent=2, ensure_ascii=False)
        except OSError:
            save_target = fallback_stmt
            with open(save_target, "w", encoding="utf-8") as f:
                json.dump(deduped, f, indent=2, ensure_ascii=False)
        return save_target, deduped
