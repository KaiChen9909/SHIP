"""Differential-privacy primitives used by SHIP.

Contents
--------
* zCDP <-> (eps, delta)-DP conversion (Theorem 2 of the paper, Bun & Steinke 2016).
* Gaussian noise / Gaussian Thresholding Mechanism for partition selection
  (Desfontaines et al., 2021; Swanberg et al., 2023).
* Privacy-budget split used by SHIP (Appendix B.1, Eq. (3)).
* Poisson subsampling and privacy amplification (Balle et al., 2018).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Hashable, Iterable, Mapping

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# zCDP -> approximate DP conversion
# --------------------------------------------------------------------------- #
def _eps_at_alpha(alpha: float, rho: float, delta: float) -> float:
    """eps(alpha) = alpha*rho + log(1 - 1/alpha) - (log(delta) + log(alpha)) / (alpha - 1)."""
    return (
        alpha * rho
        + math.log1p(-1.0 / alpha)
        - (math.log(delta) + math.log(alpha)) / (alpha - 1.0)
    )


def cdp_eps(rho: float, delta: float) -> float:
    """Smallest eps such that rho-zCDP implies (eps, delta)-DP (Theorem 2 with delta'=0).

    eps = inf_{alpha>1} [ alpha*rho + log(1-1/alpha) - (log(delta) + log(alpha)) / (alpha-1) ].
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if not 0.0 <= rho < math.inf:
        raise ValueError(f"rho must be finite and >= 0, got {rho}")
    if rho == 0:
        return 0.0
    # Coarse log-spaced search for the minimiser, then local refinement.
    grid = 1.0 + np.logspace(-6, 7, 2000)
    vals = np.array([_eps_at_alpha(a, rho, delta) for a in grid])
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    if hi > lo:
        res = minimize_scalar(
            lambda a: _eps_at_alpha(a, rho, delta), bounds=(lo, hi), method="bounded"
        )
        best = min(float(res.fun), float(vals[i]))
    else:
        best = float(vals[i])
    return max(best, 0.0)


def approx_cdp_eps(rho: float, delta: float, delta_prime: float = 0.0) -> float:
    """eps for delta'-approximate rho-zCDP -> (eps, delta)-DP, for delta > delta' (Theorem 2)."""
    if not delta > delta_prime:
        raise ValueError("Need delta > delta_prime.")
    delta_bar = (delta - delta_prime) / (1.0 - delta_prime)
    return cdp_eps(rho, delta_bar)


def cdp_rho(eps: float, delta: float) -> float:
    """Largest rho such that rho-zCDP implies (eps, delta)-DP (numerical inverse of `cdp_eps`).

    This is f_rho in Appendix B.1 of the paper.
    """
    if not 0.0 < eps < math.inf:
        raise ValueError(f"eps must be finite and > 0, got {eps}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    hi = 1.0
    while cdp_eps(hi, delta) < eps:
        hi *= 2.0
    # eps is monotone increasing in rho.
    # Include zero so small positive eps can yield rho below 1e-12. Use a relative
    # stopping criterion: a fixed absolute tolerance can overwhelm such budgets.
    rho = brentq(
        lambda r: cdp_eps(r, delta) - eps, 0.0, hi,
        xtol=np.nextafter(0.0, 1.0), maxiter=500,
    )
    # brentq may land marginally above the target; step down to be conservative.
    while cdp_eps(rho, delta) > eps:
        rho *= 1 - 1e-9
    return float(rho)


# --------------------------------------------------------------------------- #
# Gaussian mechanism / partition selection
# --------------------------------------------------------------------------- #
def gaussian_sigma(rho: float, l2_sensitivity: float = 1.0) -> float:
    """Noise scale for rho-zCDP Gaussian mechanism: sigma = Delta_2 / sqrt(2 rho)."""
    if rho <= 0:
        raise ValueError(f"rho must be > 0, got {rho}")
    return l2_sensitivity / math.sqrt(2.0 * rho)


def gaussian_threshold(sigma: float, delta: float) -> float:
    """Threshold of the Gaussian Thresholding Mechanism: tau = 1 + sigma * Phi^{-1}(1 - delta)."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    return 1.0 + sigma * float(norm.isf(delta))


def add_gaussian_noise(
    counts: Mapping[Hashable, float], sigma: float, rng: np.random.Generator
) -> dict[Hashable, float]:
    """Add i.i.d. N(0, sigma^2) noise to every entry of a histogram (including zero counts)."""
    keys = list(counts.keys())
    if sigma == 0 or not keys:
        return {k: float(counts[k]) for k in keys}
    noise = rng.normal(0.0, sigma, size=len(keys))
    return {k: float(counts[k]) + float(z) for k, z in zip(keys, noise)}


def gaussian_partition_selection(
    counts: Mapping[Hashable, int],
    sigma: float,
    threshold: float,
    rng: np.random.Generator,
) -> dict[Hashable, float]:
    """Gaussian Thresholding Mechanism (Desfontaines et al., 2021).

    `counts` must contain only *observed* partitions (count >= 1), and each record must
    contribute to at most one partition (L2 sensitivity 1). Returns {partition: noisy_count}
    for partitions whose noisy count reaches `threshold`.
    """
    noisy = add_gaussian_noise(counts, sigma, rng)
    return {k: v for k, v in noisy.items() if counts[k] > 0 and v >= threshold}


def noisy_threshold(
    counts: Mapping[Hashable, int],
    candidates: Iterable[Hashable],
    sigma: float,
    threshold: float,
    rng: np.random.Generator,
) -> tuple[dict[Hashable, float], dict[Hashable, float]]:
    """Measure every candidate (zero counts included) with Gaussian noise and threshold.

    Returns (all_noisy_counts, surviving_noisy_counts). Since the candidate domain is fixed
    by previous DP outputs, this is a plain Gaussian mechanism followed by post-processing.
    """
    full = {c: counts.get(c, 0) for c in candidates}
    noisy = add_gaussian_noise(full, sigma, rng)
    return noisy, {k: v for k, v in noisy.items() if v >= threshold}


# --------------------------------------------------------------------------- #
# Subsampling
# --------------------------------------------------------------------------- #
def poisson_subsample(n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Indices of a Poisson subsample (each record kept independently with prob. `rate`)."""
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    return np.flatnonzero(rng.random(n) < rate)


def amplified_eps_delta(eps_sub: float, delta_sub: float, rate: float) -> tuple[float, float]:
    """Global (eps, delta) of an (eps_sub, delta_sub)-DP mechanism run on a Poisson subsample."""
    return math.log1p(rate * math.expm1(eps_sub)), rate * delta_sub


def subsample_budget(eps: float, delta: float, rate: float) -> tuple[float, float]:
    """Inverse of `amplified_eps_delta`: the (eps_sub, delta_sub) allowed on the subsample
    so that the overall mechanism is (eps, delta)-DP on the full corpus."""
    eps_sub = math.log1p(math.expm1(eps) / rate)
    delta_sub = delta / rate
    if delta_sub >= 1:
        raise ValueError("delta / rate >= 1; choose a smaller delta or a larger rate.")
    return eps_sub, delta_sub


def default_delta(n: int) -> float:
    """delta = 1 / (n ln n), as in Appendix B.6."""
    if n < 2:
        raise ValueError("Need n >= 2 to compute default delta.")
    return 1.0 / (n * math.log(n))


# --------------------------------------------------------------------------- #
# SHIP budget split
# --------------------------------------------------------------------------- #
@dataclass
class SHIPBudget:
    """All privacy parameters used inside one SHIP run (on the possibly subsampled data)."""

    epsilon: float
    delta: float
    rho_total: float
    rho_tax: float
    rho_aim: float
    delta_tax: float
    num_attributes: int
    depth: int
    rho_step: float
    delta_step: float
    rho_open: float
    rho_close: float
    sigma_open: float
    sigma_close: float
    tau_open: float
    tau_close: float
    open_fraction: float
    # Filled when Poisson subsampling is used.
    global_epsilon: float | None = None
    global_delta: float | None = None
    subsample_rate: float | None = None
    budget_mode: str = "tight"
    delta_conversion: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def ship_budget(
    epsilon: float,
    delta: float,
    num_attributes: int,
    depth: int,
    tau_close: float = 10.0,
    open_fraction: float = 0.9,
    delta_tax_fraction: float = 1.0 / 3.0,
    aim_fraction: float = 0.5,
    budget_mode: str = "tight",
) -> SHIPBudget:
    """Privacy-budget split of SHIP (Appendix B.1 and Eq. (3)).

    delta_tax = delta / 3,
    rho_total = f_rho(eps, delta_conversion),
    rho_aim = rho_tax = rho_total / 2,
    rho_step = rho_tax / (|A| d),  delta_step = delta_tax / (|A| d),
    rho_open = 0.9 rho_step,  rho_close = 0.1 rho_step.

    "tight" (default) uses delta_conversion = (delta - delta_tax) / (1 - delta_tax)
    from Theorem 1. "loose" uses delta_conversion = delta - delta_tax, matching
    Appendix B.1's 2 delta / 3 when delta_tax_fraction = 1/3. The tight value is
    larger, allowing a larger rho and less noise at the same target (eps, delta).
    The loose split is conservative because
    delta_tax + (1 - delta_tax) * delta_conversion <= delta.
    """
    if not 0.0 < epsilon < math.inf:
        raise ValueError(f"epsilon must be finite and > 0, got {epsilon}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if num_attributes < 1 or depth < 1:
        raise ValueError("num_attributes and depth must be >= 1.")
    for name, fraction in (
        ("open_fraction", open_fraction),
        ("delta_tax_fraction", delta_tax_fraction),
        ("aim_fraction", aim_fraction),
    ):
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {fraction}")
    if budget_mode not in ("tight", "loose"):
        raise ValueError(f"budget_mode must be 'tight' or 'loose', got {budget_mode!r}")
    delta_tax = delta * delta_tax_fraction
    delta_conversion = delta - delta_tax
    if budget_mode == "tight":
        delta_conversion /= 1.0 - delta_tax
    rho_total = cdp_rho(epsilon, delta_conversion)
    rho_aim = aim_fraction * rho_total
    rho_tax = rho_total - rho_aim
    steps = num_attributes * depth
    rho_step, delta_step = rho_tax / steps, delta_tax / steps
    rho_open, rho_close = open_fraction * rho_step, (1.0 - open_fraction) * rho_step
    sigma_open, sigma_close = gaussian_sigma(rho_open), gaussian_sigma(rho_close)
    return SHIPBudget(
        epsilon=epsilon,
        delta=delta,
        rho_total=rho_total,
        rho_tax=rho_tax,
        rho_aim=rho_aim,
        delta_tax=delta_tax,
        num_attributes=num_attributes,
        depth=depth,
        rho_step=rho_step,
        delta_step=delta_step,
        rho_open=rho_open,
        rho_close=rho_close,
        sigma_open=sigma_open,
        sigma_close=sigma_close,
        tau_open=gaussian_threshold(sigma_open, delta_step),
        tau_close=tau_close,
        open_fraction=open_fraction,
        budget_mode=budget_mode,
        delta_conversion=delta_conversion,
    )

