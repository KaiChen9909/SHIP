"""Configuration objects for SHIP."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def prompt_path(name: str) -> str:
    return os.path.join(PROMPT_DIR, name)


@dataclass
class Attribute:
    """A semantic dimension A_i of the structured insight (output of workload analysis)."""

    name: str
    description: str
    # Optional public granularity hints (Figure 5 "Hierarchy Examples", Figure 8 variants A/B).
    example_path: str | None = None
    reference_categories: list[str] | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @staticmethod
    def from_dict(d: dict) -> "Attribute":
        return Attribute(
            name=str(d["name"]).strip(),
            description=str(d.get("description", "")).strip(),
            example_path=d.get("example_path") or None,
            reference_categories=d.get("reference_categories") or None,
        )


@dataclass
class SHIPConfig:
    # ---- privacy ----
    epsilon: float = 1.0  # global epsilon; float("inf") => non-private run
    delta: float | None = None  # global delta; default 1 / (n ln n) with n = full corpus size
    subsample_rate: float | None = None  # Poisson subsampling ratio r (Appendix B.3); None = no subsampling
    open_fraction: float = 0.9  # rho_open = 0.9 rho_step, rho_close = 0.1 rho_step
    aim_fraction: float = 0.5  # rho_aim = rho_tax = rho_total / 2
    delta_tax_fraction: float = 1.0 / 3.0  # delta_tax = delta / 3
    tau_close: float = 10.0  # closed-stage threshold (Appendix B.1 / B.6)
    nondp_open_threshold: float = 10.0  # frequency filter for open categories when epsilon = inf

    # ---- taxonomy ----
    depth: int = 3  # max hierarchy depth d
    dedup_open_before_comp: bool = False  # also de-duplicate C_open before the complementary proposal
    max_complementary: int | None = None  # optional cap on |C_comp| per parent
    max_text_chars: int | None = 20000  # truncate long records in classification prompts

    # ---- models (Gemini) ----
    attribute_model: str = "gemini-3.5-flash"  # M_attr
    taxonomy_model: str = "gemini-3.5-flash"  # M_tax (complementary proposal + de-duplication)
    classification_model: str = "gemini-3.5-flash-lite"  # M_cls (proposal + re-classification)
    temperature: float = 0.0
    thinking_budget: int | None = None
    max_workers: int = 16
    cache_path: str | None = ".cache/llm_cache.sqlite"
    vertexai: bool = False

    # ---- AIM ----
    aim_degree: int = 2
    aim_max_model_size: float = 80.0
    aim_max_iters: int = 1000
    aim_rounds: int | None = None
    num_synthetic_rows: int | None = None  # default: AIM's estimate of the record count

    # ---- misc ----
    seed: int | None = None
    save_private_allocation: bool = False  # WARNING: the allocation table is NOT DP; debugging only.

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
