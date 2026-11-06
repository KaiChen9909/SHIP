"""Taxonomy inspection and tabular cross-tabulation helpers pre-loaded into the agent Python REPL."""

from __future__ import annotations

import collections
import re
from typing import Any

import pandas as pd

_ROOT_NAMES = {"root", "<root>", ""}


def get_helper_functions_prompt_doc() -> str:
    """Return system-prompt documentation for pre-loaded Python REPL helper functions."""
    return (
        "\nGlobally Pre-loaded Helper Functions in Python Execution Environment (do NOT import them, call them directly):\n"
        "Note: Helper functions return structured dictionaries/DataFrames and do NOT print results by themselves. "
        "Always use `print(...)` to display helper function returns, variables, and computed data structures to STDOUT.\n"
        "1. `tree_inspect(category_tree, annotation_name, parent_category=None)`: Inspects taxonomy tree"
        " structure, child category levels, and full paths. Calling `print(tree_inspect(category_tree, 'annotation_name'))`"
        " returns all Level-1 (macro root) categories directly under `direct_children`!\n"
        "2. `tabular_stat(df, annotation_name, category_names=None, category_tree=None, level=None)`: Calculates record count and"
        " percentage distribution for taxonomy categories.\n"
        "   - Level-1 (Macro Root Categories): Call `print(tabular_stat(df, 'annotation_name', level=1))` to get the"
        " complete breakdown of all Level-1 macro categories, or pass root category names e.g. `tabular_stat(df, 'annotation_name', ['root_category'])`.\n"
        "   - Level-2 or Deeper Levels: Pass hierarchical paths (e.g. 'Level 1 > Level 2 > Leaf'), path prefixes"
        " (e.g. 'Level 1 > Level 2'), or single category names.\n"
        "3. `tabular_crosstab(df, annotations, categories)` / `tabular_crosstab(df, annotation_1, annotation_2, level=None)`:"
        " Calculates cross-tabulation distribution between two taxonomy annotations (supports `level=1` for Level-1 macro cross-tabulations).\n"
    )


def unwrap_annotation_tree(category_tree: dict[str, Any], annotation_name: str) -> dict[str, Any]:
    """Normalize both SHIP taxonomy.json ({'tree': {...}}) and category_tree.json ({'name': 'root', ...})."""
    node = category_tree[annotation_name]
    if isinstance(node, dict) and "tree" in node and isinstance(node["tree"], dict):
        return node["tree"]
    return node


def find_node_by_name(node: dict[str, Any], target_name: str) -> dict[str, Any] | None:
    """Recursively search for a taxonomy node matching target_name (case-insensitive)."""
    node_name = str(node.get("name") or node.get("category") or "").strip().lower()
    if node_name == target_name.strip().lower():
        return node
    for child in node.get("children", []):
        if child:
            found = find_node_by_name(child, target_name)
            if found:
                return found
    return None


def get_node_level(node: dict[str, Any], target_name: str, current_level: int = 0) -> int | None:
    """Return 1-indexed depth of target_name inside node."""
    name = str(node.get("name") or node.get("category") or "").strip().lower()
    if current_level > 0 and name == target_name.strip().lower():
        return current_level
    for child in node.get("children", []):
        if child:
            lvl = get_node_level(child, target_name, current_level + 1)
            if lvl is not None:
                return lvl
    return None


def collect_children_by_level(node: dict[str, Any], current_level: int = 0) -> dict[str, list[str]]:
    """Collect descendant category names grouped by 'level_1', 'level_2', etc."""
    level_map: dict[str, list[str]] = collections.defaultdict(list)
    for child in node.get("children", []):
        if not child:
            continue
        cname = str(child.get("name") or child.get("category") or "").strip()
        if cname:
            clvl = current_level + 1
            level_map[f"level_{clvl}"].append(cname)
            for k, v in collect_children_by_level(child, clvl).items():
                level_map[k].extend(v)
    return dict(level_map)


def collect_all_tree_nodes_with_paths(node: dict[str, Any], current_prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    """Collect every non-root node paired with its 'L1 > L2 > ...' path string."""
    results: list[tuple[str, dict[str, Any]]] = []
    node_name = str(node.get("name") or node.get("category") or "").strip()
    is_root = node_name.lower() in _ROOT_NAMES and not current_prefix
    if is_root:
        path_prefix = ""
    else:
        path_prefix = f"{current_prefix} > {node_name}" if current_prefix else node_name
        results.append((path_prefix, node))
    for child in node.get("children", []):
        if child:
            results.extend(collect_all_tree_nodes_with_paths(child, path_prefix))
    return results


def get_node_descendant_paths(node: dict[str, Any], current_prefix: str = "") -> list[str]:
    """Collect full leaf paths under node."""
    node_name = str(node.get("name") or node.get("category") or "").strip()
    is_root = node_name.lower() in _ROOT_NAMES and not current_prefix
    path_prefix = "" if is_root else (f"{current_prefix} > {node_name}" if current_prefix else node_name)
    children = [c for c in node.get("children", []) if c]
    if not children:
        return [path_prefix] if path_prefix else []
    out: list[str] = []
    for child in children:
        out.extend(get_node_descendant_paths(child, path_prefix))
    return out


def resolve_categories_to_paths(
    category_tree: dict[str, Any] | None, annotation_name: str, category_names: list[str]
) -> set[str]:
    """Expand category names or prefixes into lowercase leaf paths using category_tree when provided."""
    if not category_tree or annotation_name not in category_tree:
        return {" > ".join(s.strip().lower() for s in str(c).split(">")) for c in category_names if str(c).strip()}
    ann_tree = unwrap_annotation_tree(category_tree, annotation_name)
    all_nodes = collect_all_tree_nodes_with_paths(ann_tree)
    resolved: set[str] = set()
    for cat in category_names:
        cat_norm = " > ".join(s.strip().lower() for s in str(cat).strip().split(">"))
        matched = [(p, n) for p, n in all_nodes if p.lower().strip() == cat_norm]
        if not matched and " > " not in cat_norm:
            matched = [
                (p, n) for p, n in all_nodes
                if str(n.get("name") or n.get("category") or "").lower().strip() == cat_norm
            ]
        if matched:
            for path_prefix, node in matched:
                parts = path_prefix.split(" > ")
                parent_prefix = " > ".join(parts[:-1]) if len(parts) > 1 else ""
                resolved.update(get_node_descendant_paths(node, current_prefix=parent_prefix))
        else:
            resolved.add(cat_norm)
    return {p.lower().strip() for p in resolved if p}


def tree_inspect(
    category_tree: dict[str, Any], annotation_name: str, parent_category: str | None = None
) -> dict[str, Any]:
    """Inspect direct children and level-grouped descendants under annotation_name (and optional parent_category)."""
    if annotation_name not in category_tree:
        return {
            "error": f"Annotation '{annotation_name}' not found in category tree. Available: {list(category_tree.keys())}"
        }
    ann_tree = unwrap_annotation_tree(category_tree, annotation_name)
    if parent_category is None:
        target_node, parent_level, parent_prefix = ann_tree, 0, ""
    else:
        all_nodes = collect_all_tree_nodes_with_paths(ann_tree)
        parent_norm = " > ".join(s.strip().lower() for s in parent_category.split(">"))
        matched = [(p, n) for p, n in all_nodes if p.lower().strip() == parent_norm]
        if not matched and " > " not in parent_norm:
            matched = [
                (p, n) for p, n in all_nodes
                if str(n.get("name") or n.get("category") or "").lower().strip() == parent_norm
            ]
        if matched:
            parent_prefix, target_node = matched[0]
            parent_level = len(parent_prefix.split(" > "))
        else:
            target_node = find_node_by_name(ann_tree, parent_category)
            if not target_node:
                return {"error": f"Parent category '{parent_category}' not found under '{annotation_name}'."}
            parent_level = get_node_level(ann_tree, parent_category, 0) or 1
            parent_prefix = parent_category

    children = [c for c in target_node.get("children", []) if c]
    child_names = [str(c.get("name") or c.get("category") or "").strip() for c in children]
    child_names = [c for c in child_names if c]
    child_paths = [f"{parent_prefix} > {c}" if parent_prefix else c for c in child_names]
    return {
        "annotation_name": annotation_name,
        "parent_category": parent_category,
        "parent_level": parent_level,
        "direct_children": child_names,
        "direct_children_paths": child_paths,
        "direct_children_count": len(child_names),
        "children_by_level": collect_children_by_level(target_node, current_level=parent_level),
    }


def _resolve_column_series(df: pd.DataFrame, annotation_name: str) -> tuple[pd.Series | None, str | None]:
    """Resolve annotation_name from '<attr>', '<attr> (path)', or divided '<attr> (level k)' columns."""
    if annotation_name in df.columns:
        return df[annotation_name].fillna("").astype(str).str.strip(), annotation_name
    path_col = f"{annotation_name} (path)"
    if path_col in df.columns:
        return df[path_col].fillna("").astype(str).str.strip(), path_col
    level_cols = [c for c in df.columns if c.startswith(f"{annotation_name} (level ")]
    if level_cols:
        sorted_cols = sorted(
            level_cols,
            key=lambda c: int(m.group(1)) if (m := re.search(r"\(level\s*(\d+)\)", c)) else 999,
        )

        def _join_row(row: pd.Series) -> str:
            parts: list[str] = []
            for idx, val in enumerate(row):
                if pd.isna(val):
                    continue
                s = str(val).strip()
                if not s:
                    continue
                if idx > 0 and s.lower() == "none":
                    continue
                parts.append(s)
            return " > ".join(parts) if parts else "none"

        return df[sorted_cols].apply(_join_row, axis=1), annotation_name
    return None, None


def _build_category_mask(
    series: pd.Series, target_cats: list[str], category_tree: dict[str, Any] | None, annotation_name: str
) -> tuple[pd.Series, list[str]]:
    """Build boolean match mask supporting leaf paths, path prefixes, and individual node names."""
    clean_paths = resolve_categories_to_paths(category_tree, annotation_name, target_cats)
    s_lower = series.str.lower().str.strip()
    mask = pd.Series(False, index=series.index)
    if clean_paths:
        mask = mask | s_lower.isin(clean_paths)
    for target in target_cats:
        t_norm = " > ".join(s.strip() for s in str(target).strip().lower().split(">"))
        if not t_norm:
            continue
        mask = mask | (s_lower == t_norm) | s_lower.str.startswith(t_norm + " > ")
        if " > " not in t_norm:
            mask = mask | s_lower.apply(lambda p: t_norm in [seg.strip() for seg in p.split(">")])
    return mask, sorted(clean_paths)


def tabular_stat(
    df: pd.DataFrame,
    annotation_name: str,
    category_names: str | list[str] | None = None,
    category_name: str | list[str] | None = None,
    category_tree: dict[str, Any] | None = None,
    level: int | None = None,
) -> dict[str, Any]:
    """Compute record counts and percentages for an attribute or list of categories/paths."""
    series, _ = _resolve_column_series(df, annotation_name)
    if series is None:
        return {"error": f"Column '{annotation_name}' not in DataFrame. Available: {df.columns.tolist()}"}

    targets = category_names if category_names is not None else category_name
    total_rows = len(df)
    non_none_count = int((series.str.lower() != "none").sum())

    if targets is None:
        if level is not None and level > 0:
            idx = int(level) - 1
            lvl_col = f"{annotation_name} (level {int(level)})"
            if lvl_col in df.columns:
                lvl_series = df[lvl_col].fillna("none").astype(str).str.strip()
            else:
                lvl_series = series.apply(
                    lambda s: [p.strip() for p in s.split(">")][idx]
                    if len([p.strip() for p in s.split(">")]) > idx
                    else s
                )
            counts = lvl_series.value_counts().to_dict()
        else:
            counts = series.value_counts().to_dict()
        breakdown = [
            {
                "category": cat_val,
                "count": int(cnt),
                "pct_entire_dataset": round((cnt / total_rows) * 100.0, 2) if total_rows > 0 else 0.0,
            }
            for cat_val, cnt in counts.items()
        ]
        return {
            "annotation_name": annotation_name,
            "total_dataset_rows": total_rows,
            "non_none_rows": non_none_count,
            "categories_breakdown": breakdown,
        }

    target_list = [targets] if isinstance(targets, str) else list(targets)
    mask, mapped_paths = _build_category_mask(series, target_list, category_tree, annotation_name)
    match_count = int(mask.sum())
    pct_ds = (match_count / total_rows) * 100.0 if total_rows > 0 else 0.0
    pct_nn = (match_count / non_none_count) * 100.0 if non_none_count > 0 else 0.0
    return {
        "annotation_name": annotation_name,
        "input_categories": target_list,
        "mapped_paths": mapped_paths,
        "matching_count": match_count,
        "count": match_count,
        "total_dataset_rows": total_rows,
        "non_none_rows": non_none_count,
        "pct_entire_dataset": round(pct_ds, 2),
        "percentage": round(pct_ds, 2),
        "pct_non_none_subset": round(pct_nn, 2),
    }


def tabular_crosstab(
    df: pd.DataFrame,
    annotations: str | list[str] | None = None,
    categories: str | list[str] | list[list[str]] | None = None,
    annotation_1: str | None = None,
    annotation_2: str | None = None,
    category_tree: dict[str, Any] | None = None,
    level: int | None = None,
) -> dict[str, Any]:
    """Compute 2-attribute full contingency tables or N-attribute conditioned joint counts."""
    if annotation_1 is not None and annotation_2 is not None:
        ann_list = [annotation_1, annotation_2]
    elif isinstance(annotations, str) and (isinstance(categories, str) or categories is None):
        ann_list = [annotations, str(categories)] if categories else [annotations]
    elif isinstance(annotations, list):
        ann_list = annotations
    else:
        ann_list = []

    if len(ann_list) == 2 and (categories is None or isinstance(categories, str)):
        col1, col2 = ann_list
        s1, _ = _resolve_column_series(df, col1)
        s2, _ = _resolve_column_series(df, col2)
        if s1 is None or s2 is None:
            return {"error": f"Columns missing from DataFrame. Available: {df.columns.tolist()}"}
        if level is not None and level > 0:
            idx = int(level) - 1
            s1 = s1.apply(lambda s: [p.strip() for p in s.split(">")][idx] if len([p.strip() for p in s.split(">")]) > idx else s)
            s2 = s2.apply(lambda s: [p.strip() for p in s.split(">")][idx] if len([p.strip() for p in s.split(">")]) > idx else s)
        ct = pd.crosstab(s1, s2)
        return {
            "annotation_1": col1,
            "annotation_2": col2,
            "total_dataset_rows": len(df),
            "crosstab_counts": {str(r): {str(c): int(v) for c, v in row.items()} for r, row in ct.iterrows()},
        }

    if not isinstance(annotations, list) or not isinstance(categories, list) or len(annotations) != len(categories):
        return {"error": "Provide matching lists for `annotations` and `categories`."}

    masks: list[pd.Series] = []
    mapped_by_ann: list[list[str]] = []
    for ann, cat_item in zip(annotations, categories):
        ser, _ = _resolve_column_series(df, ann)
        if ser is None:
            return {"error": f"Column '{ann}' missing from DataFrame. Available: {df.columns.tolist()}"}
        cats = [cat_item] if isinstance(cat_item, str) else [str(x) for x in cat_item]
        col_mask, mapped = _build_category_mask(ser, cats, category_tree, ann)
        masks.append(col_mask)
        mapped_by_ann.append(mapped)

    joint_mask = masks[0].copy()
    for m in masks[1:]:
        joint_mask = joint_mask & m
    joint_count = int(joint_mask.sum())
    total_rows = len(df)
    indiv = {ann: int(m.sum()) for ann, m in zip(annotations, masks)}
    cond = {
        f"pct_joint_given_{ann}": round((joint_count / cnt) * 100.0, 2) if cnt > 0 else 0.0
        for ann, cnt in indiv.items()
    }
    return {
        "annotations": annotations,
        "input_categories": categories,
        "mapped_paths_by_annotation": mapped_by_ann,
        "individual_counts": indiv,
        "joint_matching_count": joint_count,
        "total_dataset_rows": total_rows,
        "conditional_probabilities": cond,
        "pct_joint_overall": round((joint_count / total_rows) * 100.0, 2) if total_rows > 0 else 0.0,
    }
