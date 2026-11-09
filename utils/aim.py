"""AIM: An Adaptive and Iterative Mechanism for DP synthetic data (McKenna et al., 2024).

Adapted from `private-pgm/mechanisms/aim.py` (https://github.com/ryan112358/private-pgm) and
parameterised directly by a zCDP budget `rho`, so it composes with SHIP's accounting.
Requires the `mbi` package (private-pgm, JAX backend). Works with both the 1.x API
(`estimation.mirror_descent(..., potentials=...)`) and the >= 2.0 API
(`estimation.MirrorDescent().estimate(..., warm_start=...)`); requirements.txt pins a tested commit.
"""

from __future__ import annotations

import itertools
import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import softmax

logger = logging.getLogger(__name__)

Clique = tuple[str, ...]


def _powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s) + 1))


def _downward_closure(cliques) -> list[Clique]:
    ans: set[Clique] = set()
    for cl in cliques:
        ans.update(_powerset(cl))
    return sorted(ans, key=len)


def _compile_workload(workload: Sequence[tuple[Clique, float]]) -> dict[Clique, float]:
    weights = {cl: wt for cl, wt in workload}

    def score(cl):
        return sum(w * len(set(cl) & set(wcl)) for wcl, w in weights.items())

    return {cl: score(cl) for cl in _downward_closure(weights.keys())}


def _filter_candidates(candidates, model, size_limit):
    from mbi import junction_tree

    free = set(_downward_closure(model.cliques))
    return {
        cl: w
        for cl, w in candidates.items()
        if junction_tree.hypothetical_model_size(model.domain, list(model.cliques) + [cl]) <= size_limit
        or cl in free
    }


def _estimate(domain, measurements, iters: int, warm_start=None):
    """Mirror-descent estimation of a graphical model, across private-pgm API versions."""
    from mbi import estimation

    if hasattr(estimation, "MirrorDescent"):  # mbi >= 2.0
        return estimation.MirrorDescent().estimate(domain, measurements, iters=iters, warm_start=warm_start)
    return estimation.mirror_descent(  # mbi 1.x
        domain, measurements, iters=iters, potentials=warm_start, callback_fn=lambda *_: None
    )


def _exponential_mechanism(qualities: dict, epsilon: float, sensitivity: float, rng: np.random.Generator):
    keys = list(qualities.keys())
    q = np.array([qualities[k] for k in keys], dtype=float)
    q = q - q.max()
    p = softmax(0.5 * epsilon / sensitivity * q)
    return keys[rng.choice(len(keys), p=p)]


def aim(
    df: pd.DataFrame,
    domain: Mapping[str, int],
    rho: float,
    workload: Sequence[tuple[Clique, float]] | None = None,
    degree: int = 2,
    rounds: int | None = None,
    max_model_size: float = 80.0,
    max_iters: int = 1000,
    num_rows: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Run AIM under rho-zCDP.

    Args:
        df: integer-coded table; column `c` takes values in {0, ..., domain[c]-1}.
        domain: {column: domain size}.
        rho: zCDP budget for the whole mechanism.
        workload: list of (clique, weight). Defaults to all `degree`-way marginals, weight 1.
        rounds: max number of adaptive rounds (default 16 * #columns).
        max_model_size: size limit (MB) of the graphical model.
        max_iters: mirror-descent iterations per estimation.
        num_rows: rows of synthetic data (default: the model's estimated total).
        rng: numpy Generator for the DP noise.

    Returns:
        Integer-coded synthetic DataFrame with the same columns.
    """
    from mbi import Dataset, Domain, LinearMeasurement

    rng = rng or np.random.default_rng()
    cols = list(domain.keys())
    if rho <= 0:
        raise ValueError("AIM needs rho > 0.")
    dom = Domain(cols, [int(domain[c]) for c in cols])
    data = Dataset({c: df[c].to_numpy(dtype=int) for c in cols}, dom)

    if workload is None:
        k = min(degree, len(cols))
        workload = [(cl, 1.0) for cl in itertools.combinations(cols, k)]
    rounds = rounds or 16 * len(cols)
    candidates = _compile_workload(workload)
    answers = {cl: np.asarray(data.project(cl).datavector()) for cl in candidates}

    oneway = [cl for cl in candidates if len(cl) == 1]
    sigma = np.sqrt(rounds / (2 * 0.9 * rho))
    epsilon = np.sqrt(8 * 0.1 * rho / rounds)

    def noisy(cl, s):
        x = answers[cl]
        return LinearMeasurement(x + rng.normal(0, s, x.size), cl, stddev=s)

    measurements = [noisy(cl, sigma) for cl in oneway]
    rho_used = len(oneway) * 0.5 / sigma**2
    model = _estimate(dom, measurements, max_iters)
    potentials = None

    t, terminate = 0, False
    while not terminate:
        t += 1
        if rho - rho_used < 2 * (0.5 / sigma**2 + 1.0 / 8 * epsilon**2):
            remaining = rho - rho_used
            sigma = np.sqrt(1 / (2 * 0.9 * remaining))
            epsilon = np.sqrt(8 * 0.1 * remaining)
            terminate = True
        rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
        size_limit = max_model_size * rho_used / rho

        small = _filter_candidates(candidates, model, size_limit)
        errors = {}
        for cl, wgt in small.items():
            bias = np.sqrt(2 / np.pi) * sigma * dom.size(cl)
            xest = np.asarray(model.project(cl).datavector())
            errors[cl] = wgt * (np.linalg.norm(answers[cl] - xest, 1) - bias)
        max_sens = max(abs(w) for w in small.values())
        cl = _exponential_mechanism(errors, epsilon, max_sens, rng)
        logger.info("AIM round %d: measuring %s (rho used %.4g / %.4g)", t, cl, rho_used, rho)

        n = dom.size(cl)
        measurements.append(noisy(cl, sigma))
        z = np.asarray(model.project(cl).datavector())
        pcliques = list({m.clique for m in measurements})
        potentials = model.potentials.expand(pcliques)
        model = _estimate(dom, measurements, max_iters, warm_start=potentials)
        w = np.asarray(model.project(cl).datavector())
        if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
            sigma /= 2
            epsilon *= 2

    model = _estimate(dom, measurements, max_iters, warm_start=potentials)
    synth = model.synthetic_data(rows=num_rows).to_dict()
    return pd.DataFrame({c: np.asarray(synth[c]).astype(int) for c in cols})
