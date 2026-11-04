"""Hierarchical Hybrid DP Taxonomy Building (Algorithm 1 / Algorithm 3 of the paper).

All attributes are processed jointly in each record-level LLM call (as in the paper's
implementation, Appendix F), but every attribute has its own histograms and noise, so the
privacy accounting of Algorithm 2 (rho_step per attribute per level) applies unchanged.

Per level l = 1..d and per active parent c_p (for every attribute):
  1. Discovery: M_cls freely proposes y_i for every record under c_p.        (Fig. 5)
  2. Open partition selection: Gaussian thresholding over (c_p, y) bins     -> C_open
  3. Complementary proposal: M_tax proposes C_comp given C_open (public).    (Fig. 7)
  4. Re-classification of records with y_i not in C_open over C_open ∪ C_comp. (Fig. 6)
  5. Closed thresholding: noisy counts of C_open ∪ C_comp, keep >= tau_close -> C_surv
     (records of filtered categories are mapped to "other").
  6. De-duplication / refinement of C_surv with M_tax (post-processing).     (Fig. 8)
  7. Attach C_surv ∪ {other, none} as children of c_p; extend record paths.

Privacy notes: every step that touches records is record-wise; the set of parents that is
processed, the complementary proposals and the de-duplication depend only on earlier DP
outputs and public inputs. We therefore never skip a parent based on (private) record counts.
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

_NONE_ALIASES = {"none", "n/a", "na", "null", "not applicable", "unspecified", "not specified", ""}
_OTHER_ALIASES = {"other", "others", "miscellaneous", "misc"}


def normalize_category(value: Any) -> str:
    """Canonical surface form of a category name; maps placeholder variants to other/none."""
    if value is None:
        return NONE
    if isinstance(value, (list, tuple)):
        value = value[0] if value else NONE
    s = str(value).strip().lower()
    s = s.strip("\"'`*.,;: ")
    s = s.replace(">", "-").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    if s in _NONE_ALIASES:
        return NONE
    if s in _OTHER_ALIASES:
        return OTHER
    return s


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
        return [
            p for p in self.nodes if len(p) == depth and (depth == 0 or p[-1] not in RESERVED)
        ]

    def leaves(self) -> list[Path]:
        """Root-to-leaf paths: the categorical domain of this attribute."""
        return [p for p, n in self.nodes.items() if p and not n.children]

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
        """Open discovery for one record, all of its active attributes in one call (Figure 5)."""
        prompt = render_prompt(
            prompt_path("propose_categories.txt"),
            dataset_description=self.desc, max_depth=self.d, current_level=level,
            annotations="\n\n".join(self._annotation_block(k + 1, a, p) for k, (a, p) in enumerate(items)),
            text=truncate(text, self.cfg.max_text_chars),
        )
        schema = self._object_schema({a: {"type": "string"} for a, _ in items})
        obj = self._json_call(prompt, self.cfg.classification_model, schema)
        return {a: normalize_category(self._lookup(obj, a)) for a, _ in items}

    def _reclassify(
        self, text: str, items: list[tuple[str, Path, list[str]]], level: int
    ) -> dict[str, str]:
        """Closed re-classification for one record, all attributes needing it in one call (Figure 6).
        The schema restricts each answer to its candidates plus other/none."""
        prompt = render_prompt(
            prompt_path("reclassify.txt"),
            dataset_description=self.desc, max_depth=self.d, current_level=level,
            annotations="\n\n".join(
                self._annotation_block(k + 1, a, p, cands) for k, (a, p, cands) in enumerate(items)
            ),
            text=truncate(text, self.cfg.max_text_chars),
        )
        schema = self._object_schema(
            {a: {"type": "string", "enum": list(cands) + list(RESERVED)} for a, _, cands in items}
        )
        obj = self._json_call(prompt, self.cfg.classification_model, schema)
        out = {}
        for a, _, cands in items:
            y = normalize_category(self._lookup(obj, a))
            out[a] = y if (y in cands or y in RESERVED) else OTHER
        return out

    def _complementary(self, attr: str, parent: Path, passed: list[str]) -> list[str]:
        """Complementary proposal with the taxonomy model (Figure 7). Public inputs only."""
        a = self.attr_by_name[attr]
        prompt = render_prompt(
            prompt_path("complementary_categories.txt"),
            dataset_description=self.desc, annotation_name=attr, description=a.description,
            parent_path=self._parent(parent), passed_categories=self._bullets(passed),
        )
        schema = {"type": "array", "items": {"type": "string"}}
        obj = self._json_call(prompt, self.cfg.taxonomy_model, schema)
        banned = set(passed) | set(RESERVED) | set(parent)
        out: list[str] = []
        for c in obj:
            c = normalize_category(c)
            if c and c not in banned and c not in out:
                out.append(c)
        if self.cfg.max_complementary is not None:
            out = out[: self.cfg.max_complementary]
        return out

    def _deduplicate(self, attr: str, parent: Path, cats: list[str]) -> dict[str, str]:
        """LLM de-duplication of surviving categories (Figure 8). Post-processing only."""
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
        schema = self._object_schema({c: {"type": "string"} for c in cats})
        obj = self._json_call(prompt, self.cfg.taxonomy_model, schema)
        mapping = {}
        for c in cats:
            v = self._lookup(obj, c)
            v = normalize_category(v) if v is not None else c
            mapping[c] = v if v not in parent else c  # never collapse into an ancestor name
        return mapping

    # ---- main loop ------------------------------------------------------- #
    def build(self, texts: list[str]) -> tuple[dict[str, AttributeTaxonomy], dict[str, list[Path]]]:
        n = len(texts)
        names = [a.name for a in self.attributes]
        paths: dict[str, list[Path]] = {a: [()] * n for a in names}
        b = self.budget

        for level in range(1, self.d + 1):
            parents = {a: self.taxonomies[a].expandable(level - 1) for a in names}
            parent_sets = {a: set(ps) for a, ps in parents.items()}
            # (record, attribute) pairs still being refined at this level.
            active: dict[int, list[str]] = defaultdict(list)
            for a in names:
                for i, p in enumerate(paths[a]):
                    if len(p) == level - 1 and p in parent_sets[a]:
                        active[i].append(a)
            logger.info("Level %d: %d parents, %d active records",
                        level, sum(map(len, parents.values())), len(active))

            # 1. Discovery (open proposals).
            rec_ids = sorted(active)
            props = thread_map(
                lambda i: self._propose(texts[i], [(a, paths[a][i]) for a in active[i]], level),
                rec_ids, self.cfg.max_workers, desc=f"L{level} propose",
            )
            y: dict[str, dict[int, str]] = {a: {} for a in names}
            for i, res in zip(rec_ids, props):
                for a, c in res.items():
                    y[a][i] = c

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

            # 2b. (optional) De-duplicate C_open before the complementary proposal.
            if self.cfg.dedup_open_before_comp:
                for a in names:
                    jobs = [(p, cs) for p, cs in c_open[a].items() if len(cs) > 1]
                    maps = thread_map(lambda pc: self._deduplicate(a, *pc), jobs, self.cfg.max_workers,
                                      desc=f"L{level} dedup-open {a}", show_progress=False)
                    for (p, cs), m in zip(jobs, maps):
                        merged: dict[str, float] = {}
                        for c in cs:
                            merged[m[c]] = merged.get(m[c], 0.0) + open_noisy[a].pop((p, c))
                        open_noisy[a].update({(p, c): v for c, v in merged.items() if c not in RESERVED})
                        c_open[a][p] = [c for c in merged if c not in RESERVED]
                        for i, c in y[a].items():
                            if paths[a][i] == p and c in m:
                                y[a][i] = m[c]

            # 3. Complementary proposal for every active parent (data-independent given C_open).
            jobs = [(a, p) for a in names for p in parents[a]]
            comps = thread_map(lambda ap: self._complementary(ap[0], ap[1], c_open[ap[0]][ap[1]]),
                               jobs, self.cfg.max_workers, desc=f"L{level} complementary")
            c_comp = {a: {} for a in names}
            for (a, p), cs in zip(jobs, comps):
                c_comp[a][p] = cs
            candidates = {a: {p: c_open[a][p] + c_comp[a][p] for p in parents[a]} for a in names}

            # 4. Re-classify records whose proposal did not survive open selection.
            re_items: dict[int, list[tuple[str, Path, list[str]]]] = defaultdict(list)
            for a in names:
                open_sets = {p: set(cs) for p, cs in c_open[a].items()}
                for i, c in y[a].items():
                    p = paths[a][i]
                    if c == NONE or c in open_sets[p]:
                        continue
                    if candidates[a][p]:
                        re_items[i].append((a, p, candidates[a][p]))
                    else:
                        y[a][i] = OTHER
            re_ids = sorted(re_items)
            reclass = thread_map(lambda i: self._reclassify(texts[i], re_items[i], level),
                                 re_ids, self.cfg.max_workers, desc=f"L{level} reclassify")
            for i, res in zip(re_ids, reclass):
                for a, c in res.items():
                    y[a][i] = c

            # 5. Closed thresholding over C_open ∪ C_comp (every candidate measured, zeros included).
            c_surv: dict[str, dict[Path, list[str]]] = {a: {} for a in names}
            close_noisy: dict[str, dict[tuple[Path, str], float]] = {a: {} for a in names}
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

            # 6. De-duplicate / refine surviving categories.
            jobs = [(a, p) for a in names for p in parents[a] if len(c_surv[a][p]) > 1]
            maps = thread_map(lambda ap: self._deduplicate(ap[0], ap[1], c_surv[ap[0]][ap[1]]),
                              jobs, self.cfg.max_workers, desc=f"L{level} dedup")
            dedup = {a: {} for a in names}
            for (a, p), m in zip(jobs, maps):
                dedup[a][p] = m

            # 7. Attach children and extend record paths.
            for a in names:
                tax = self.taxonomies[a]
                open_set = {(p, c) for (p, c) in open_noisy[a]}
                for p in parents[a]:
                    m = dedup[a].get(p, {c: c for c in c_surv[a][p]})
                    for c in c_surv[a][p]:
                        canon = m.get(c, c)
                        if canon in RESERVED:
                            continue
                        is_open = (p, c) in open_set
                        node = tax.add_child(p, canon, source="open" if is_open else "closed")
                        if is_open:  # a merged node counts as open if any member was discovered
                            node.source = "open"
                            node.noisy_count_open = (node.noisy_count_open or 0.0) + open_noisy[a][(p, c)]
                        node.noisy_count_closed = (node.noisy_count_closed or 0.0) + close_noisy[a][(p, c)]
                    tax.add_child(p, OTHER, source="reserved")
                    tax.add_child(p, NONE, source="reserved")
                for i, c in y[a].items():
                    p = paths[a][i]
                    if c not in RESERVED:
                        c = dedup[a].get(p, {}).get(c, c)
                    paths[a][i] = p + (c,)

                self.private_diagnostics.append(self._diag(a, level, parents[a], c_open[a], c_comp[a], y[a], paths[a]))
                logger.info("Level %d / %s: |C_open|=%d |C_comp|=%d children=%d", level, a,
                            sum(map(len, c_open[a].values())), sum(map(len, c_comp[a].values())),
                            sum(len(tax.nodes[p].children) for p in parents[a]))

        return self.taxonomies, paths

    def _diag(self, a, level, parents, c_open, c_comp, y, paths) -> dict:
        tax = self.taxonomies[a]
        src = Counter()
        for i in y:
            node = tax.nodes[paths[i]]
            src[node.source] += 1
        return {
            "attribute": a,
            "level": level,
            "num_parents": len(parents),
            "num_open_selected": sum(map(len, c_open.values())),
            "num_complementary": sum(map(len, c_comp.values())),
            "num_children": sum(len(tax.nodes[p].children) for p in parents),
            "records_by_source": dict(src),
        }
