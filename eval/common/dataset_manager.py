"""Dataset context discovery, divided-column preparation, and prompt/workload loaders."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import pandas as pd

from .agent_functions import unwrap_annotation_tree
from .sandbox import ALLOWED_DATASET_FILENAMES

DATASET_DESCRIPTIONS: dict[str, str] = {
    "wildchat": (
        "Conversations between human users and AI chatbots across diverse topics, "
        "languages, task intents, formatting constraints, and assistant response strategies."
    ),
    "chatdoctor": (
        "Medical consultations between patients describing symptoms and chief complaints "
        "and doctors recommending diagnostic examinations and provisional diagnoses."
    ),
    "cfpb": (
        "Consumer financial complaints submitted to the CFPB describing banking, credit, "
        "billing, or loan grievances and requested company resolutions."
    ),
}


def safe_format_prompt(template_str: str, **kwargs: Any) -> str:
    """Substitute {key} placeholders without escaping literal JSON braces inside prompt templates."""
    out = template_str
    for k, v in kwargs.items():
        out = out.replace("{" + str(k) + "}", str(v))
    return out


def ensure_hierarchical_compat_files(data_dir: str) -> None:
    """Ensure category_tree.json and explicit '<attr> (level k)' CSV columns exist in data_dir."""
    tax_path = os.path.join(data_dir, "taxonomy.json")
    cat_tree_path = os.path.join(data_dir, "category_tree.json")
    if os.path.isfile(tax_path) and not os.path.isfile(cat_tree_path):
        with open(tax_path, encoding="utf-8") as f:
            raw_tax = json.load(f)
        unwrapped = {}
        for attr, val in raw_tax.items():
            tree = unwrap_annotation_tree(raw_tax, attr)
            unwrapped[attr] = {"name": "root", "children": tree.get("children", [])}
        with open(cat_tree_path, "w", encoding="utf-8") as f:
            json.dump(unwrapped, f, indent=2, ensure_ascii=False)

    # Open-source SHIP's synthetic.csv already has '<attr> (level 1..d)' and '<attr> (path)'.
    if os.path.isfile(os.path.join(data_dir, "synthetic.csv")):
        return

    for base_name in ("synthetic_allocated_paths", "allocated_paths"):
        src_csv = os.path.join(data_dir, f"{base_name}.csv")
        dst_csv = os.path.join(data_dir, f"{base_name}_divided.csv")
        if not os.path.isfile(src_csv) or os.path.isfile(dst_csv):
            continue
        df = pd.read_csv(src_csv)
        new_cols: dict[str, pd.Series] = {}
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(100)
            if not any(" > " in v for v in sample):
                new_cols[col] = df[col]
                continue
            split_lists = df[col].fillna("none").astype(str).apply(lambda s: [p.strip() for p in s.split(">")])
            max_d = max((len(x) for x in split_lists), default=1)
            for lvl in range(1, max_d + 1):
                new_cols[f"{col} (level {lvl})"] = split_lists.apply(
                    lambda parts, idx=lvl - 1: parts[idx] if idx < len(parts) else "none"
                )
        pd.DataFrame(new_cols).to_csv(dst_csv, index=False)


class DatasetContext:
    """Resolved file paths and metadata for a single hierarchical dataset directory."""

    def __init__(self, algo_name: str, data_dir: str, intro_text: str = ""):
        self.algo_name = algo_name
        self.data_dir = os.path.abspath(data_dir)
        self.intro_text = intro_text
        ensure_hierarchical_compat_files(self.data_dir)
        self.data_files = sorted(
            f for f in os.listdir(self.data_dir)
            if os.path.isfile(os.path.join(self.data_dir, f)) and f.lower() in ALLOWED_DATASET_FILENAMES
        )

    def get_id(self) -> str:
        digest = hashlib.md5(self.data_dir.encode("utf-8")).hexdigest()[:8]
        return f"{self.algo_name}_{digest}"

    @property
    def main_csv_path(self) -> str:
        """Primary CSV path for Closed-Ended QA (prefers divided level columns)."""
        for cand in (
            "synthetic.csv",
            "synthetic_allocated_paths_divided.csv",
            "allocated_paths_divided.csv",
            "synthetic_allocated_paths.csv",
            "allocated_paths.csv",
        ):
            p = os.path.join(self.data_dir, cand)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No recognized CSV found in {self.data_dir} (checked synthetic.csv, *allocated_paths*.csv).")

    @property
    def full_path_csv_path(self) -> str:
        """Primary CSV path for Open-Ended QA (prefers full path columns or synthetic.csv)."""
        for cand in (
            "synthetic.csv",
            "synthetic_allocated_paths.csv",
            "allocated_paths.csv",
            "synthetic_allocated_paths_divided.csv",
            "allocated_paths_divided.csv",
        ):
            p = os.path.join(self.data_dir, cand)
            if os.path.isfile(p):
                return p
        return self.main_csv_path

    @property
    def category_tree_path(self) -> str:
        for cand in ("category_tree.json", "taxonomy.json"):
            p = os.path.join(self.data_dir, cand)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"Neither category_tree.json nor taxonomy.json found in {self.data_dir}.")

    @property
    def dataset_info_path(self) -> str:
        return os.path.join(self.data_dir, "dataset_info.json")


class DatasetTemplateManager:
    """Factory helpers for DatasetContext, workload dicts, and JSON prompt templates."""

    @staticmethod
    def load_dataset(
        algo_name: str, data_dir: str, dataset_name: str = "wildchat", dataset_description: str = ""
    ) -> DatasetContext:
        intro = dataset_description.strip()
        if not intro:
            info_path = os.path.join(data_dir, "dataset_info.json")
            if os.path.isfile(info_path):
                with open(info_path, encoding="utf-8") as f:
                    info = json.load(f)
                intro = str(info.get("dataset_description") or info.get("description") or "").strip()
        if not intro:
            intro = DATASET_DESCRIPTIONS.get(dataset_name.lower(), f"Dataset '{dataset_name}'.")
        return DatasetContext(algo_name=algo_name, data_dir=data_dir, intro_text=intro)

    @staticmethod
    def load_workload_dict(workload_file_path: str) -> dict[str, dict[str, str]]:
        """Load workload JSON into normalized {key: {'title': ..., 'description': ...}} form."""
        with open(workload_file_path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return {str(i + 1): {"title": f"Query {i + 1}", "description": str(q)} for i, q in enumerate(raw)}
        if isinstance(raw, dict) and "queries" in raw and isinstance(raw["queries"], list):
            return {str(i + 1): {"title": f"Query {i + 1}", "description": str(q)} for i, q in enumerate(raw["queries"])}
        out: dict[str, dict[str, str]] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                out[str(k)] = {
                    "title": str(v.get("title", f"Query {k}")),
                    "description": str(v.get("description", "")),
                }
            else:
                out[str(k)] = {"title": f"Query {k}", "description": str(v)}
        return out

    @staticmethod
    def load_prompt_template(prompt_path: str) -> str:
        """Read a JSON prompt file ({'prompt_template': [...]}) and join lines into a template string."""
        with open(prompt_path, encoding="utf-8") as f:
            obj = json.load(f)
        tmpl = obj.get("prompt_template", "")
        return "\n".join(tmpl) if isinstance(tmpl, list) else str(tmpl)
