"""SHIP end-to-end (Algorithm 2).

    A   <- M_attr(W)                                   (workload analysis, public)
    rho_step, delta_step via Eq. (3)
    for A in A: T_A, {y_i^(A)} <- Algorithm 1/3        (DP taxonomy + allocation)
    T   <- union_A T_A
    D~  <- AIM({(y_i^(A))_A}, T, rho_aim)              (DP tabular synthesis)
    return T, D~

Privacy (Theorem 1): the release (T, D~) is (eps, delta)-DP with
rho = rho_tax + rho_aim and delta_bar = (delta - delta_tax) / (1 - delta_tax).
The default "tight" budget mode uses delta_bar for conversion; "loose" uses
delta - delta_tax, the conservative split from Appendix B.1.
If `subsample_rate` is set, SHIP runs on a Poisson subsample with the amplified-back
budget (eps_sub, delta_sub), so the global guarantee is (eps, delta) on the full corpus.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.dp import (
    SHIPBudget,
    approx_cdp_eps,
    default_delta,
    poisson_subsample,
    ship_budget,
    subsample_budget,
)
from utils.io import save_json
from utils.llm import GeminiClient

from .attributes import determine_attributes, hydrate_example_paths
from .config import Attribute, SHIPConfig
from .synthesis import paths_to_frame, synthesize
from .taxonomy import AttributeTaxonomy, HierarchicalTaxonomyBuilder

logger = logging.getLogger(__name__)


@dataclass
class SHIPResult:
    attributes: list[Attribute]
    taxonomies: dict[str, AttributeTaxonomy]
    synthetic: pd.DataFrame
    budget: SHIPBudget
    # ---- internal / NOT differentially private ----
    private_allocation: pd.DataFrame | None = None
    private_diagnostics: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def taxonomy_dict(self) -> dict:
        return {a: t.to_dict() for a, t in self.taxonomies.items()}

    def save(self, output_dir: str, save_private: bool = False, export_eval_compat: bool = True) -> None:
        os.makedirs(output_dir, exist_ok=True)
        save_json([a.to_dict() for a in self.attributes], os.path.join(output_dir, "attributes.json"))
        save_json(self.taxonomy_dict(), os.path.join(output_dir, "taxonomy.json"))
        if export_eval_compat:
            save_json({a: t.to_category_tree_dict() for a, t in self.taxonomies.items()},
                      os.path.join(output_dir, "category_tree.json"))
        save_json(self.budget.to_dict(), os.path.join(output_dir, "privacy.json"))
        save_json(self.usage, os.path.join(output_dir, "usage.json"))
        self.synthetic.to_csv(os.path.join(output_dir, "synthetic.csv"), index=False)
        if save_private and self.private_allocation is not None:
            pdir = os.path.join(output_dir, "private_DO_NOT_RELEASE")
            os.makedirs(pdir, exist_ok=True)
            self.private_allocation.to_csv(os.path.join(pdir, "allocation.csv"), index=False)
            save_json(self.private_diagnostics, os.path.join(pdir, "diagnostics.json"))


class SHIP:
    def __init__(self, cfg: SHIPConfig, llm: GeminiClient | None = None):
        self.cfg = cfg
        self.llm = llm or GeminiClient(cache_path=cfg.cache_path, vertexai=cfg.vertexai)
        self.rng = np.random.default_rng(cfg.seed)

    def _budget(self, n_full: int, num_attributes: int) -> SHIPBudget:
        cfg = self.cfg
        if not (0.0 < cfg.epsilon < math.inf):
            raise ValueError(f"epsilon must be a finite positive number, got {cfg.epsilon}")
        delta = cfg.delta if cfg.delta is not None else default_delta(n_full)
        eps_run, delta_run = cfg.epsilon, delta

        # if subsampling is used, compute the amplified-back budget for the run.
        if cfg.subsample_rate is not None and cfg.subsample_rate < 1.0:
            eps_run, delta_run = subsample_budget(cfg.epsilon, delta, cfg.subsample_rate)

        budget = ship_budget(
            eps_run, delta_run, num_attributes, cfg.depth, tau_close=cfg.tau_close,
            open_fraction=cfg.open_fraction, delta_tax_fraction=cfg.delta_tax_fraction,
            aim_fraction=cfg.aim_fraction,
            budget_mode=cfg.budget_mode,
        )
        budget.global_epsilon, budget.global_delta = cfg.epsilon, delta
        budget.subsample_rate = cfg.subsample_rate

        # Sanity check of Theorem 1 on the run-level budget.
        achieved = approx_cdp_eps(budget.rho_tax + budget.rho_aim, delta_run, budget.delta_tax)
        assert achieved <= eps_run * (1 + 1e-6), (achieved, eps_run)
        return budget

    def run(
        self,
        texts: list[str],
        workload: list[str],
        dataset_description: str,
        attributes: list[Attribute] | None = None,
    ) -> SHIPResult:
        cfg, t0 = self.cfg, time.time()
        n_full = len(texts)

        # Workload analysis and granularity calibration (public: uses only W and S_D).
        if attributes is None:
            attributes = determine_attributes(self.llm, workload, dataset_description, cfg)
        else:
            attributes = hydrate_example_paths(self.llm, attributes, dataset_description, cfg)
        logger.info("Attributes: %s", [a.name for a in attributes])

        budget = self._budget(n_full, len(attributes))
        logger.info("Privacy budget: %s", budget.to_dict())

        if cfg.subsample_rate is not None and cfg.subsample_rate < 1.0:
            idx = poisson_subsample(n_full, cfg.subsample_rate, self.rng)
            texts = [texts[i] for i in idx]
            logger.info("Poisson subsample: %d / %d records (r=%g)", len(texts), n_full, cfg.subsample_rate)

        # DP taxonomy building + allocation.
        builder = HierarchicalTaxonomyBuilder(self.llm, attributes, dataset_description, budget, cfg, self.rng)
        taxonomies, paths = builder.build(texts)

        # DP tabular synthesis.
        synthetic = synthesize(taxonomies, paths, budget.rho_aim, cfg, self.rng)

        usage = self.llm.usage.to_dict() if hasattr(self.llm, "usage") else {}
        usage["wall_time_s"] = round(time.time() - t0, 1)
        return SHIPResult(
            attributes=attributes,
            taxonomies=taxonomies,
            synthetic=synthetic,
            budget=budget,
            private_allocation=paths_to_frame(paths, cfg.depth),
            private_diagnostics=builder.private_diagnostics,
            usage=usage,
        )
