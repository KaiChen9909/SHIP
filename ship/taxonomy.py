"""Hierarchical Hybrid DP Taxonomy Building (Algorithm 1 / Algorithm 3 of the paper).

Record-level LLM calls handle all of a record's still-active attributes at once; every attribute
keeps its own histograms and noise, so the accounting of Algorithm 2 (rho_step per attribute per
level) applies unchanged. An attribute whose path has ended (reserved leaf, or a parent that
stopped expanding) is not sent again at deeper levels.

Per level l = 1..d, for every attribute and every expandable parent c_p:
  1. Discovery: M_cls freely proposes y_i for every record under c_p.            (propose_categories)
  2. Open partition selection: Gaussian thresholding over (c_p, y) bins     -> C_open
     2a. (level 1) LLM maps null-like survivors ("no constraints", ...) to none. (identify_none_categories)
     2b. De-duplication of C_open (chunked for large sets).                      (deduplicate_categories)
  3. Complementary proposal: M_tax proposes C_comp given C_open.                (complementary_categories)
  4. Re-classification over C_open ∪ C_comp of records whose proposal did not
     survive, including records that proposed "none".                           (reclassify)
  5. Closed thresholding: noisy counts of C_open ∪ C_comp, keep >= tau_close -> C_surv.
  6. (optional) De-duplication of C_surv (Algorithm 3 line 25).
  7. Attach C_surv plus the reserved fallbacks (other / none at the root, "other <parent>" /
     "general <parent>" below it) under c_p. A non-root parent with no surviving category stays a
     leaf. Extend record paths.
After all levels: whole-tree disambiguation of names repeated across branches.  (disambiguate_tree_names)

Privacy: steps 1, 4 and the counts in 2 and 5 are record-wise; everything else (2a, 2b, 3, 6, 7,
disambiguation) depends only on DP outputs and public inputs. Control flow never depends on
private quantities such as how many records reach a parent.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.dp import SHIPBudget, gaussian_partition_selection, noisy_threshold
from utils.io import render_prompt, truncate
from utils.llm import GeminiClient, thread_map

from .config import Attribute, SHIPConfig, prompt_path

logger = logging.getLogger(__name__)

OTHER, NONE = "other", "none"
RESERVED = (OTHER, NONE)
PATH_SEP = " > "
Path = tuple[str, ...]

_NONE_ALIASES = {
    "", "none", "n/a", "na", "null", "unknown", "not applicable", "unspecified", "not specified",
    "none specified", "no constraints", "no constraint", "unconstrained",
    "no format constraints", "no formatting constraints", "no specific constraint",
}
_OTHER_ALIASES = {"other", "others", "miscellaneous", "misc"}
_STRIP = "\"'`*.,;:- "

STRING_LIST_SCHEMA = {"type": "array", "items": {"type": "string"}}
DEDUP_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"raw": {"type": "string"}, "canonical": {"type": "string"}},
        "required": ["raw", "canonical"],
    },
}
DISAMBIGUATE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "new_name": {"type": "string"}},
        "required": ["path", "new_name"],
    },
}


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
def normalize_category(value: Any) -> str:
    """Canonical surface form of a category name; maps placeholder variants to other/none."""
    if value is None:
        return NONE
    if isinstance(value, (list, tuple)):
        value = value[0] if value else NONE
    s = str(value).strip().lower().strip(_STRIP)
    for prefix in ("new:", "category:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip().strip(_STRIP)
    s = re.sub(r"\s+", " ", s.replace(">", "-"))
    if s in _NONE_ALIASES:
        return NONE
    if s in _OTHER_ALIASES:
        return OTHER
    return s


def clean_and_match(raw: Any, candidates: list[str]) -> str:
    """Map a free-text re-classification answer onto the candidates (or other / none)."""
    val = normalize_category(raw)
    if val in RESERVED:
        return val
    by_lower = {c.lower(): c for c in candidates}
    if val in by_lower:
        return by_lower[val]
    contained = [c for c in candidates if c.lower() in val]  # answer adds words around a candidate
    if contained:
        return max(contained, key=len)
    containing = [c for c in candidates if len(val) >= 3 and val in c.lower()]  # abbreviated answer
    return containing[0] if len(containing) == 1 else OTHER


def other_label(parent: Path) -> str:
    """Name of the reserved "other" child: categories too rare to survive DP selection."""
    return f"other {parent[-1]}" if parent else OTHER


def none_label(parent: Path) -> str:
    """Name of the reserved "none" child: no information beyond the parent."""
    return f"general {parent[-1]}" if parent else NONE


def format_path(path: Path) -> str:
    return PATH_SEP.join(path)


# --------------------------------------------------------------------------- #
# Taxonomy data structure
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    path: Path
    source: str  # "root" | "open" | "closed" | "reserved"
    noisy_count_open: float | None = None  # released by Gaussian thresholding (open stage)
    noisy_count_closed: float | None = None  # released by the closed-stage Gaussian measurement
    children: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path[-1] if self.path else "<root>"

    @property
    def reserved(self) -> bool:
        return self.source == "reserved"


class AttributeTaxonomy:
    """Tree-shaped domain Omega_A of one attribute."""

    def __init__(self, attribute: Attribute):
        self.attribute = attribute
        self.nodes: dict[Path, Node] = {(): Node((), "root")}

    def add_child(self, parent: Path, name: str, **kw) -> Node:
        path = parent + (name,)
        if path not in self.nodes:
            self.nodes[path] = Node(path, **kw)
            self.nodes[parent].children.append(name)
        return self.nodes[path]

    def expandable(self, depth: int) -> list[Path]:
        """Non-reserved nodes at `depth` (root for depth 0): the parents processed next."""
        return [p for p, n in self.nodes.items() if len(p) == depth and not n.reserved]

    def leaves(self) -> list[Path]:
        """Root-to-leaf paths: the categorical domain of this attribute."""
        return [p for p, n in self.nodes.items() if p and not n.children]

    def rename_node(self, old_path: Path, new_name: str) -> dict[Path, Path]:
        """Rename the last element of `old_path`, moving its subtree. Reserved children named after
        the node ("other x" / "general x") follow the new name. Returns {old path: new path};
        empty if the rename is a no-op or would collide with a sibling."""
        if not old_path or old_path not in self.nodes:
            return {}
        parent, old_name = old_path[:-1], old_path[-1]
        new_path = parent + (new_name,)
        if new_name == old_name or new_path in self.nodes:
            return {}
        node = self.nodes[old_path]
        regular = {c for c in node.children if not self.nodes[old_path + (c,)].reserved}
        relabel = {
            old: new
            for old, new in ((other_label(old_path), other_label(new_path)),
                             (none_label(old_path), none_label(new_path)))
            if old in node.children and self.nodes[old_path + (old,)].reserved and new not in regular
        }

        siblings = self.nodes[parent].children
        siblings[siblings.index(old_name)] = new_name
        k = len(old_path)
        remap: dict[Path, Path] = {}
        for p in [p for p in self.nodes if p[:k] == old_path]:
            tail = p[k:]
            if tail and tail[0] in relabel:
                tail = (relabel[tail[0]],) + tail[1:]
            remap[p] = new_path + tail
        moved = {remap[p]: self.nodes.pop(p) for p in remap}
        for p, n in moved.items():
            n.path = p
        moved[new_path].children = [relabel.get(c, c) for c in moved[new_path].children]
        self.nodes.update(moved)
        return remap

    def to_dict(self) -> dict:
        def rec(path: Path) -> dict:
            n = self.nodes[path]
            out: dict[str, Any] = {"name": n.name, "source": n.source}
            if n.noisy_count_open is not None:
                out["noisy_count_open"] = round(n.noisy_count_open, 3)
            if n.noisy_count_closed is not None:
                out["noisy_count_closed"] = round(n.noisy_count_closed, 3)
            if n.children:
                out["children"] = [rec(path + (c,)) for c in n.children]
            return out

        return {"attribute": self.attribute.to_dict(), "tree": rec(())}

    def to_category_tree_dict(self) -> dict:
        """Tree in the `category_tree.json` schema shared with ground-truth directories (eval/)."""

        def rec(path: Path) -> dict:
            n = self.nodes[path]
            count = n.noisy_count_closed if n.noisy_count_closed is not None else n.noisy_count_open
            children = [rec(path + (c,)) for c in n.children]
            return {
                "name": n.name if path else "Root",
                "is_leaf": not children,
                "type": n.source,
                "dp_count": round(count or 0.0, 3),
                "children": children,
            }

        return rec(())

    @staticmethod
    def from_dict(d: dict) -> "AttributeTaxonomy":
        tax = AttributeTaxonomy(Attribute.from_dict(d["attribute"]))

        def rec(node: dict, path: Path):
            for ch in node.get("children", []):
                tax.add_child(
                    path, ch["name"], source=ch["source"],
                    noisy_count_open=ch.get("noisy_count_open"),
                    noisy_count_closed=ch.get("noisy_count_closed"),
                )
                rec(ch, path + (ch["name"],))

        rec(d["tree"], ())
        return tax


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
class HierarchicalTaxonomyBuilder:
    def __init__(
        self,
        llm: GeminiClient,
        attributes: list[Attribute],
        dataset_description: str,
        budget: SHIPBudget,
        cfg: SHIPConfig,
        rng: np.random.Generator,
    ):
        self.llm = llm
        self.attributes = attributes
        self.attr_by_name = {a.name: a for a in attributes}
        self.desc = dataset_description
        self.budget = budget
        self.cfg = cfg
        self.rng = rng
        self.d = cfg.depth
        self.taxonomies = {a.name: AttributeTaxonomy(a) for a in attributes}
        # Internal (private) diagnostics, e.g. Table 13-style statistics. NOT DP.
        self.private_diagnostics: list[dict] = []

    # ---- LLM helpers ----------------------------------------------------- #
    def _json_call(self, prompt: str, model: str, schema: dict) -> Any:
        """Structured-output call; returns an empty container of the schema's type on failure."""
        kind = list if schema["type"] == "array" else dict
        try:
            return self.llm.generate_json(
                prompt, model, schema=schema, temperature=self.cfg.temperature,
                thinking_budget=self.cfg.thinking_budget,
                validator=lambda o: isinstance(o, kind),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM call failed permanently (%s); using fallback.", e)
            return kind()

    @staticmethod
    def _lookup(obj: dict, name: str) -> Any:
        if name in obj:
            return obj[name]
        low = {str(k).strip().lower(): v for k, v in obj.items()}
        return low.get(name.lower())

    @staticmethod
    def _object_schema(props: dict[str, dict]) -> dict:
        return {"type": "object", "properties": props, "required": list(props)}

    @staticmethod
    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {c}" for c in items) if items else "(none yet)"

    @staticmethod
    def _parent(path: Path) -> str:
        return format_path(path) if path else "(root: top-level categories)"

    def _example(self, attr: str) -> str:
        return self.attr_by_name[attr].example_path or "(not provided)"

    def _annotation_block(self, i: int, attr: str, parent: Path, candidates: list[str] | None = None) -> str:
        a = self.attr_by_name[attr]
        lines = [
            f'Annotation {i}: "{attr}"',
            f"Description: {a.description}",
            f"Parent Category Allocation Path for this Annotation: {self._parent(parent)}",
        ]
        if a.example_path:
            lines.append(f"Hierarchy Example (for granularity control only): {a.example_path}")
        if candidates is not None:
            lines += ["Candidate Categories:", self._bullets(candidates)]
        return "\n".join(lines)

    def _propose(self, text: str, items: list[tuple[str, Path]], level: int) -> dict[str, str]:
        """Open discovery for one record over its active attributes (Figure 5).
        Falls back to one call per attribute if the joint call fails."""
        prompt = render_prompt(
            prompt_path("propose_categories.txt"),
            dataset_description=self.desc, max_depth=self.d, current_level=level,
            annotations="\n\n".join(self._annotation_block(k + 1, a, p) for k, (a, p) in enumerate(items)),
            text=truncate(text, self.cfg.max_text_chars),
        )
        schema = self._object_schema({a: {"type": "string"} for a, _ in items})
        obj = self._json_call(prompt, self.cfg.classification_model, schema)
        if not obj and len(items) > 1:
            out: dict[str, str] = {}
            for item in items:
                out.update(self._propose(text, [item], level))
            return out
        return {a: normalize_category(self._lookup(obj, a)) for a, _ in items}

    def _reclassify(self, text: str, items: list[tuple[str, Path, list[str]]], level: int) -> dict[str, str]:
        """Closed re-classification for one record over several attributes (Figure 6).
        Small option sets are enum-constrained; large ones use free text + `clean_and_match`."""
        prompt = render_prompt(
            prompt_path("reclassify.txt"),
            dataset_description=self.desc, max_depth=self.d, current_level=level,
            annotations="\n\n".join(
                self._annotation_block(k + 1, a, p, cands) for k, (a, p, cands) in enumerate(items)
            ),
            text=truncate(text, self.cfg.max_text_chars),
        )
        n_options = sum(len(cands) + len(RESERVED) for _, _, cands in items)
        if n_options <= self.cfg.reclassify_enum_max:
            props = {a: {"type": "string", "enum": list(cands) + list(RESERVED)} for a, _, cands in items}
        else:
            props = {a: {"type": "string"} for a, _, _ in items}
        obj = self._json_call(prompt, self.cfg.classification_model, self._object_schema(props))
        if not obj and len(items) > 1:
            out: dict[str, str] = {}
            for item in items:
                out.update(self._reclassify(text, [item], level))
            return out
        return {a: clean_and_match(self._lookup(obj, a), cands) for a, _, cands in items}

    def _identify_none_variants(self, root_open: dict[str, list[str]]) -> dict[str, set[str]]:
        """Level-1 null-like categories among DP survivors (one call for all attributes)."""
        queried = [a for a, cats in root_open.items() if cats]
        out: dict[str, set[str]] = {a: set() for a in root_open}
        if not queried:
            return out
        blocks = "\n\n".join(
            f'Attribute: "{a}" (Description: {self.attr_by_name[a].description})\n'
            f"Discovered Categories:\n{self._bullets(root_open[a])}"
            for a in queried
        )
        prompt = render_prompt(prompt_path("identify_none_categories.txt"), attribute_blocks=blocks)
        schema = self._object_schema({a: STRING_LIST_SCHEMA for a in queried})
        obj = self._json_call(prompt, self.cfg.taxonomy_model, schema)
        for a in queried:
            vals = self._lookup(obj, a)
            if isinstance(vals, list):
                out[a] = {normalize_category(v) for v in vals} & set(root_open[a])
        return out

    def _complementary(self, attr: str, parent: Path, passed: list[str]) -> list[str]:
        """Complementary proposal with the taxonomy model (Figure 7). Public inputs + DP C_open only."""
        a = self.attr_by_name[attr]
        prompt = render_prompt(
            prompt_path("complementary_categories.txt"),
            dataset_description=self.desc, annotation_name=attr, description=a.description,
            parent_path=self._parent(parent), passed_categories=self._bullets(passed),
            example_path=self._example(attr),
        )
        obj = self._json_call(prompt, self.cfg.taxonomy_model, STRING_LIST_SCHEMA)
        banned = set(passed) | set(RESERVED) | set(parent) | {other_label(parent), none_label(parent)}
        out: list[str] = []
        for c in obj:
            c = normalize_category(c)
            if c and c not in banned and c not in out:
                out.append(c)
        if self.cfg.max_complementary is not None:
            out = out[: self.cfg.max_complementary]
        return out

    def _deduplicate_batch(self, attr: str, parent: Path, cats: list[str]) -> dict[str, str]:
        """One de-duplication call (Figure 8) with a list-of-pairs schema (no dynamic keys)."""
        if len(cats) < 2:
            return {c: c for c in cats}
        a = self.attr_by_name[attr]
        if a.example_path:
            reference = a.example_path
        elif a.reference_categories:
            reference = ", ".join(a.reference_categories)
        else:
            reference = "(not provided; keep the granularity of the discovered categories)"
        prompt = render_prompt(
            prompt_path("deduplicate_categories.txt"),
            annotation_name=attr, description=a.description, parent_path=self._parent(parent),
            granularity_reference=reference, discovered_categories=self._bullets(cats),
        )
        arr = self._json_call(prompt, self.cfg.taxonomy_model, DEDUP_SCHEMA)
        raw_to_canon = {
            normalize_category(it["raw"]): normalize_category(it["canonical"])
            for it in arr if isinstance(it, dict) and "raw" in it and "canonical" in it
        }
        bad = set(RESERVED) | set(parent) | {other_label(parent), none_label(parent)}
        mapping = {}
        for c in cats:
            canon = raw_to_canon.get(c, c)
            mapping[c] = c if (not canon or canon in bad) else canon
        return mapping

    def _deduplicate(self, attr: str, parent: Path, cats: list[str]) -> dict[str, str]:
        """De-duplication of a category list; lists above `dedup_chunk_size` are split into chunks
        and the resulting canonical names are consolidated in a further pass."""
        chunk = max(20, self.cfg.dedup_chunk_size)
        if len(cats) <= chunk:
            return self._deduplicate_batch(attr, parent, cats)
        combined: dict[str, str] = {}
        for s in range(0, len(cats), chunk):
            combined.update(self._deduplicate_batch(attr, parent, cats[s : s + chunk]))
        canon = sorted(set(combined.values()))
        if 1 < len(canon) < len(cats):
            second = self._deduplicate(attr, parent, canon)
            combined = {c: second.get(v, v) for c, v in combined.items()}
        return combined

    def _dedup_parents(self, attr: str, groups: dict[Path, list[str]], desc: str) -> dict[Path, dict[str, str]]:
        jobs = [(p, cs) for p, cs in groups.items() if len(cs) > 1]
        maps = thread_map(lambda pc: self._deduplicate(attr, *pc), jobs, self.cfg.max_workers,
                          desc=desc, show_progress=False)
        return {p: m for (p, _), m in zip(jobs, maps)}

    def _disambiguate_tree_names(self, attr: str, paths: list[Path], max_passes: int = 3) -> list[Path]:
        """Give identical names in different branches unique names (post-processing of the DP tree)."""
        tax = self.taxonomies[attr]
        a = self.attr_by_name[attr]
        for _ in range(max_passes):
            by_name: dict[str, list[Path]] = defaultdict(list)
            for p, node in tax.nodes.items():
                if p and not node.reserved:
                    by_name[node.name].append(p)
            dups = {name: ps for name, ps in by_name.items() if len(ps) > 1}
            if not dups:
                break
            lines, by_str = [], {}
            for name, ps in dups.items():
                lines.append(f"Duplicate Category Name: '{name}'\nHierarchical Paths:")
                for k, p in enumerate(ps, 1):
                    lines.append(f"  {k}. {format_path(p)}")
                    by_str[format_path(p)] = p
                lines.append("")
            prompt = render_prompt(
                prompt_path("disambiguate_tree_names.txt"),
                annotation_name=attr, description=a.description,
                dataset_description=self.desc, duplicate_blocks="\n".join(lines),
            )
            arr = self._json_call(prompt, self.cfg.taxonomy_model, DISAMBIGUATE_SCHEMA)
            total: dict[Path, Path] = {}  # original path -> current path, over this pass
            for it in arr:
                if not isinstance(it, dict):
                    continue
                old = by_str.get(str(it.get("path", "")).strip().lower())
                new_name = normalize_category(it.get("new_name"))
                if old is None or new_name in RESERVED:
                    continue
                remap = tax.rename_node(total.get(old, old), new_name)
                if remap:
                    total = {k: remap.get(v, v) for k, v in total.items()}
                    total.update({k: v for k, v in remap.items() if k not in total})
            if not total:
                break
            paths = [total.get(p, p) for p in paths]
        return paths

    # ---- main loop ------------------------------------------------------- #
    def build(self, texts: list[str]) -> tuple[dict[str, AttributeTaxonomy], dict[str, list[Path]]]:
        n = len(texts)
        names = [a.name for a in self.attributes]
        paths: dict[str, list[Path]] = {a: [()] * n for a in names}
        b = self.budget

        for level in range(1, self.d + 1):
            parents = {a: self.taxonomies[a].expandable(level - 1) for a in names}
            if not any(parents.values()):  # depends only on DP outputs
                break
            parent_sets = {a: set(ps) for a, ps in parents.items()}
            # (record, attribute) pairs still being refined at this level.
            active: dict[int, list[str]] = defaultdict(list)
            for a in names:
                for i, p in enumerate(paths[a]):
                    if len(p) == level - 1 and p in parent_sets[a]:
                        active[i].append(a)
            logger.info("Level %d: %d parents, %d active records",
                        level, sum(map(len, parents.values())), len(active))

            # 1. Discovery (open proposals), one call per record.
            rec_ids = sorted(active)
            props = thread_map(
                lambda i: self._propose(texts[i], [(a, paths[a][i]) for a in active[i]], level),
                rec_ids, self.cfg.max_workers, desc=f"L{level} propose",
            )
            y: dict[str, dict[int, str]] = {a: {} for a in names}
            for i, res in zip(rec_ids, props):
                for a, c in res.items():
                    p = paths[a][i]
                    # A proposal equal to a reserved label of this parent means other / none.
                    y[a][i] = OTHER if c == other_label(p) else NONE if c == none_label(p) else c

            # 2. Open partition selection (Gaussian thresholding over (parent, category) bins).
            c_open: dict[str, dict[Path, list[str]]] = {}
            open_noisy: dict[str, dict[tuple[Path, str], float]] = {}
            for a in names:
                counts = Counter((paths[a][i], c) for i, c in y[a].items() if c not in RESERVED)
                sel = gaussian_partition_selection(counts, b.sigma_open, b.tau_open, self.rng)
                open_noisy[a] = sel
                c_open[a] = {p: [] for p in parents[a]}
                for (p, c), _ in sorted(sel.items(), key=lambda kv: -kv[1]):
                    c_open[a][p].append(c)

            # 2a. Level 1: map null-like survivors to none (post-processing of DP outputs).
            if level == 1 and self.cfg.normalize_none_variants:
                nulls = self._identify_none_variants({a: c_open[a][()] for a in names})
                for a in names:
                    if not nulls[a]:
                        continue
                    c_open[a][()] = [c for c in c_open[a][()] if c not in nulls[a]]
                    for c in nulls[a]:
                        open_noisy[a].pop(((), c), None)
                    for i, c in y[a].items():
                        if c in nulls[a]:
                            y[a][i] = NONE

            # 2b. De-duplicate C_open before the complementary proposal.
            if self.cfg.dedup_open_before_comp:
                for a in names:
                    maps = self._dedup_parents(a, c_open[a], f"L{level} dedup-open {a}")
                    for p, m in maps.items():
                        merged: dict[str, float] = {}
                        for c in c_open[a][p]:
                            merged[m[c]] = merged.get(m[c], 0.0) + open_noisy[a].pop((p, c))
                        open_noisy[a].update({(p, c): v for c, v in merged.items()})
                        c_open[a][p] = list(merged)
                    for i, c in y[a].items():
                        m = maps.get(paths[a][i])
                        if m and c in m:
                            y[a][i] = m[c]

            # 3. Complementary proposal for every active parent (data-independent given C_open).
            jobs = [(a, p) for a in names for p in parents[a]]
            comps = thread_map(lambda ap: self._complementary(ap[0], ap[1], c_open[ap[0]][ap[1]]),
                               jobs, self.cfg.max_workers, desc=f"L{level} complementary")
            c_comp: dict[str, dict[Path, list[str]]] = {a: {} for a in names}
            for (a, p), cs in zip(jobs, comps):
                c_comp[a][p] = cs
            candidates = {a: {p: c_open[a][p] + c_comp[a][p] for p in parents[a]} for a in names}

            # 4. Re-classify records whose proposal did not survive open selection
            #    (including "none": the record may fit a concrete candidate once it sees them).
            re_items: dict[int, list[tuple[str, Path, list[str]]]] = defaultdict(list)
            for a in names:
                open_sets = {p: set(cs) for p, cs in c_open[a].items()}
                for i, c in y[a].items():
                    p = paths[a][i]
                    if c in open_sets[p]:
                        continue
                    if candidates[a][p]:
                        re_items[i].append((a, p, candidates[a][p]))
                    elif c != NONE:
                        y[a][i] = OTHER
            re_ids = sorted(re_items)
            reclass = thread_map(lambda i: self._reclassify(texts[i], re_items[i], level),
                                 re_ids, self.cfg.max_workers, desc=f"L{level} reclassify")
            for i, res in zip(re_ids, reclass):
                for a, c in res.items():
                    y[a][i] = c

            # 5. Closed thresholding over C_open ∪ C_comp (every candidate measured, zeros included).
            c_surv: dict[str, dict[Path, list[str]]] = {a: {} for a in names}
            close_noisy: dict[str, dict[tuple[Path, str], float]] = {}
            for a in names:
                counts = Counter((paths[a][i], c) for i, c in y[a].items() if c not in RESERVED)
                cand_bins = [(p, c) for p in parents[a] for c in candidates[a][p]]
                noisy_all, surv = noisy_threshold(counts, cand_bins, b.sigma_close, b.tau_close, self.rng)
                close_noisy[a] = noisy_all
                for p in parents[a]:
                    c_surv[a][p] = [c for c in candidates[a][p] if (p, c) in surv]
                surv_sets = {p: set(cs) for p, cs in c_surv[a].items()}
                for i, c in y[a].items():
                    if c not in RESERVED and c not in surv_sets[paths[a][i]]:
                        y[a][i] = OTHER

            # 6. (optional) De-duplicate surviving categories (Algorithm 3 line 25).
            dedup: dict[str, dict[Path, dict[str, str]]] = {a: {} for a in names}
            if self.cfg.dedup_after_close:
                for a in names:
                    dedup[a] = self._dedup_parents(a, c_surv[a], f"L{level} dedup-close {a}")

            # 7. Attach children and extend record paths. A non-root parent with no surviving
            #    category stays a leaf (a decision based on DP outputs only).
            for a in names:
                tax = self.taxonomies[a]
                expanded: set[Path] = set()
                for p in parents[a]:
                    if p and not c_surv[a][p]:
                        continue
                    expanded.add(p)
                    m = dedup[a].get(p, {})
                    for c in c_surv[a][p]:
                        is_open = (p, c) in open_noisy[a]
                        node = tax.add_child(p, m.get(c, c), source="open" if is_open else "closed")
                        if is_open:  # a merged node counts as open if any member was discovered
                            node.source = "open"
                            node.noisy_count_open = (node.noisy_count_open or 0.0) + open_noisy[a][(p, c)]
                        node.noisy_count_closed = (node.noisy_count_closed or 0.0) + close_noisy[a][(p, c)]
                    tax.add_child(p, other_label(p), source="reserved")
                    tax.add_child(p, none_label(p), source="reserved")
                for i, c in y[a].items():
                    p = paths[a][i]
                    if p not in expanded:
                        continue
                    if c == OTHER:
                        c = other_label(p)
                    elif c == NONE:
                        c = none_label(p)
                    else:
                        c = dedup[a].get(p, {}).get(c, c)
                    paths[a][i] = p + (c,)

                self.private_diagnostics.append(self._diag(a, level, parents[a], c_open[a], c_comp[a], y[a], paths[a]))
                logger.info("Level %d / %s: |C_open|=%d |C_comp|=%d expanded parents=%d/%d", level, a,
                            sum(map(len, c_open[a].values())), sum(map(len, c_comp[a].values())),
                            len(expanded), len(parents[a]))

        # 8. Whole-tree name disambiguation (post-processing).
        if self.cfg.disambiguate_tree_names:
            for a in names:
                paths[a] = self._disambiguate_tree_names(a, paths[a])

        return self.taxonomies, paths

    def _diag(self, a, level, parents, c_open, c_comp, y, paths) -> dict:
        tax = self.taxonomies[a]
        src = Counter(tax.nodes[paths[i]].source for i in y)
        return {
            "attribute": a,
            "level": level,
            "num_parents": len(parents),
            "num_open_selected": sum(map(len, c_open.values())),
            "num_complementary": sum(map(len, c_comp.values())),
            "num_children": sum(len(tax.nodes[p].children) for p in parents),
            "records_by_source": dict(src),
        }
