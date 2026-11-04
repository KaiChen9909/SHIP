"""Workload analysis (Algorithm 2, line 1): A <- M_attr(W).

The paper does not include the attribute-determination prompt, so
`prompts/attribute_determination.txt` ships empty and must be filled in before use
(or attributes can be supplied directly, see `load_attributes`).

Template variables available to `attribute_determination.txt` (Jinja2):
    workload             str        - the analytical queries, one numbered query per line
    dataset_description  str        - public description S_D of the corpus
    max_depth            int        - taxonomy depth d

The model output is constrained (Gemini structured output, `ATTRIBUTE_SCHEMA`) to a JSON list of
    {"name": "<snake_case>", "description": "<what the attribute captures>",
     "example_path": "<optional granularity example, e.g. 'stem > mathematics > calculus'>"}
"""

from __future__ import annotations

import json
import re

from utils.io import render_prompt
from utils.llm import GeminiClient

from .config import Attribute, SHIPConfig, prompt_path


def _snake(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return s or "attribute"


def parse_attributes(obj) -> list[Attribute]:
    if isinstance(obj, dict):
        obj = obj.get("attributes", obj.get("features", []))
    if not isinstance(obj, list) or not obj:
        raise ValueError("Attribute model output must be a non-empty JSON list.")
    attrs, seen = [], set()
    for d in obj:
        a = Attribute.from_dict(d)
        a.name = _snake(a.name)
        if a.name in seen:
            continue
        seen.add(a.name)
        attrs.append(a)
    return attrs


ATTRIBUTE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "example_path": {"type": "string"},
        },
        "required": ["name", "description"],
    },
}


def determine_attributes(
    llm: GeminiClient, workload: list[str], dataset_description: str, cfg: SHIPConfig
) -> list[Attribute]:
    """Map the workload to the attributes needed to answer it (no private data is accessed)."""
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
    return parse_attributes(obj)


def load_attributes(path: str) -> list[Attribute]:
    with open(path, encoding="utf-8") as f:
        return parse_attributes(json.load(f))
