"""Workload analysis (Algorithm 2, line 1): A <- M_attr(W), plus public granularity calibration.

Everything here uses only public inputs (workload W, dataset description S_D), so it has no
privacy cost.

Template variables of `attribute_determination.txt`:
    workload             str  - the analytical queries, one numbered query per line
    dataset_description  str  - public description S_D of the corpus
    max_depth            int  - taxonomy depth d
The output is constrained to `ATTRIBUTE_SCHEMA`.

Attribute inputs (`--attributes` or workload analysis) only need names and descriptions.
Files may be a JSON list of {"name", "description"} or
{"domain_annotations": {name: {"description": ...}, ...}}. By default,
`hydrate_example_paths` generates the hierarchy examples separately with the taxonomy
model, one level per call (prompt `sample_path.txt`), using public information only.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from utils.io import render_prompt
from utils.llm import GeminiClient

from .config import Attribute, SHIPConfig, prompt_path

logger = logging.getLogger(__name__)

ATTRIBUTE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["name", "description"],
    },
}

SAMPLE_PATH_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}},
    "required": ["category"],
}


def _snake(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return s or "attribute"


def parse_attributes(obj) -> list[Attribute]:
    """Parse attributes from a JSON list, {"attributes": [...]}, or {"domain_annotations": {...}}."""
    if isinstance(obj, dict):
        if isinstance(obj.get("domain_annotations"), dict):
            obj = [
                {"name": k, **v} if isinstance(v, dict) else {"name": k, "description": str(v)}
                for k, v in obj["domain_annotations"].items()
            ]
        else:
            obj = obj.get("attributes", obj.get("features", []))
    if not isinstance(obj, list) or not obj:
        raise ValueError("Attributes must be a non-empty list or a 'domain_annotations' mapping.")
    attrs, seen = [], set()
    for d in obj:
        a = Attribute.from_dict(d)
        a.name = _snake(a.name)
        if a.name in seen:
            continue
        seen.add(a.name)
        attrs.append(a)
    return attrs


def generate_sample_path(llm: GeminiClient, attr: Attribute, dataset_description: str, cfg: SHIPConfig) -> str:
    """Generate one public example path sequentially, requesting one category per level."""
    levels: list[str] = []

    def valid_category(obj: Any) -> bool:
        if not isinstance(obj, dict) or not isinstance(obj.get("category"), str):
            return False
        category = obj["category"].strip().lower()
        return (
            bool(category)
            and category not in levels
            and category not in {"none", "other", "unspecified", "not applicable", "miscellaneous", "n/a"}
            and not any(separator in category for separator in (">", "\n", "\r"))
        )

    try:
        for current_level in range(1, cfg.depth + 1):
            parent_path = " > ".join(f"{name} (level {i + 1})" for i, name in enumerate(levels))
            prompt = render_prompt(
                prompt_path("sample_path.txt"),
                dataset_description=dataset_description,
                annotation_name=attr.name,
                description=attr.description,
                max_depth=cfg.depth,
                current_level=current_level,
                parent_path=parent_path or "(root: no parent category yet)",
            )
            obj = llm.generate_json(
                prompt,
                cfg.taxonomy_model,
                schema=SAMPLE_PATH_LEVEL_SCHEMA,
                temperature=cfg.temperature,
                thinking_budget=cfg.thinking_budget,
                validator=valid_category,
            )
            levels.append(obj["category"].strip().lower())
        return " > ".join(f"{lv} (level {k + 1})" for k, lv in enumerate(levels))
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to generate a sample path for %s at level %d/%d: %s",
                       attr.name, len(levels) + 1, cfg.depth, e)
        return ""


def hydrate_example_paths(
    llm: GeminiClient, attributes: list[Attribute], dataset_description: str, cfg: SHIPConfig
) -> list[Attribute]:
    """Generate missing hierarchy examples with the LLM (enabled by default; public inputs only)."""
    if cfg.auto_example_path:
        for a in attributes:
            if not a.example_path:
                a.example_path = generate_sample_path(llm, a, dataset_description, cfg) or None
    return attributes


def determine_attributes(
    llm: GeminiClient, workload: list[str], dataset_description: str, cfg: SHIPConfig
) -> list[Attribute]:
    """Determine attribute names/descriptions, then generate their hierarchy examples."""
    prompt = render_prompt(
        prompt_path("attribute_determination.txt"),
        workload="\n".join(f"{k + 1}. {q}" for k, q in enumerate(workload)),
        dataset_description=dataset_description,
        max_depth=cfg.depth,
    )
    obj = llm.generate_json(
        prompt,
        cfg.attribute_model,
        schema=ATTRIBUTE_SCHEMA,
        temperature=cfg.temperature,
        thinking_budget=cfg.thinking_budget,
        validator=lambda o: isinstance(o, (list, dict)),
    )
    return hydrate_example_paths(llm, parse_attributes(obj), dataset_description, cfg)


def load_attributes(path: str) -> list[Attribute]:
    with open(path, encoding="utf-8") as f:
        return parse_attributes(json.load(f))
