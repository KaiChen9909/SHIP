"""Private tabular synthesis (Algorithm 2, line 6): D~ <- AIM({y_i}, T, rho_aim).

Each attribute becomes one categorical column whose domain is the set of root-to-leaf
paths of its taxonomy (data-independent given T). AIM is run on the integer-coded table and
the synthetic paths are expanded into per-level columns for release, e.g.
    "topic (level 1)", "topic (level 2)", "topic (level 3)", "topic (path)".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.aim import aim

from .config import SHIPConfig
from .taxonomy import PATH_SEP, AttributeTaxonomy, Path


def build_domains(taxonomies: dict[str, AttributeTaxonomy]) -> dict[str, list[Path]]:
    return {a: sorted(t.leaves()) for a, t in taxonomies.items()}


def encode(paths: dict[str, list[Path]], domains: dict[str, list[Path]]) -> pd.DataFrame:
    cols = {}
    for a, ps in paths.items():
        index = {p: k for k, p in enumerate(domains[a])}
        missing = {p for p in ps if p not in index}
        if missing:
            raise ValueError(f"Attribute {a}: {len(missing)} record paths not in the taxonomy domain, e.g. {next(iter(missing))}")
        cols[a] = np.fromiter((index[p] for p in ps), dtype=np.int64, count=len(ps))
    return pd.DataFrame(cols)


def decode(codes: pd.DataFrame, domains: dict[str, list[Path]], depth: int) -> pd.DataFrame:
    """Integer codes -> released table with per-level columns and the full path."""
    out = {}
    for a, dom in domains.items():
        ps = [dom[int(k)] for k in codes[a].to_numpy()]
        for lvl in range(depth):
            out[f"{a} (level {lvl + 1})"] = [p[lvl] if lvl < len(p) else None for p in ps]
        out[f"{a} (path)"] = [PATH_SEP.join(p) for p in ps]
    return pd.DataFrame(out)


def paths_to_frame(paths: dict[str, list[Path]], depth: int) -> pd.DataFrame:
    """Allocation table (NOT private) in the same layout as the released synthetic table."""
    domains = {a: sorted(set(ps)) for a, ps in paths.items()}
    return decode(encode(paths, domains), domains, depth)


def synthesize(
    taxonomies: dict[str, AttributeTaxonomy],
    paths: dict[str, list[Path]],
    rho_aim: float,
    cfg: SHIPConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run AIM with budget rho_aim on the private allocation and return the decoded synthetic table."""
    domains = build_domains(taxonomies)
    codes = encode(paths, domains)
    sizes = {a: len(d) for a, d in domains.items()}
    synth = aim(
        codes, sizes, rho_aim,
        degree=cfg.aim_degree, rounds=cfg.aim_rounds, max_model_size=cfg.aim_max_model_size,
        max_iters=cfg.aim_max_iters, num_rows=cfg.num_synthetic_rows, rng=rng,
    )
    return decode(synth, domains, cfg.depth)
