"""Command-line entry point.

Example (from the repository root):
    python -m ship.run \
        --data data/chatdoctor.jsonl --text-field text \
        --workload workloads/chatdoctor/topic_workload.json \
        --description "Conversations between patients and doctors" \
        --epsilon 1.0 --output-dir outputs/chatdoctor_eps1

    # skip workload analysis by giving attributes directly:
        --attributes workloads/chatdoctor/topic_attributes.json
"""

from __future__ import annotations

import argparse
import logging

from utils.io import load_records, load_workload, read_text_or_literal

from .attributes import load_attributes
from .config import SHIPConfig
from .pipeline import SHIP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SHIP (Structured and Hierarchical Insight with Privacy).")
    p.add_argument("--data", required=True, help=".jsonl/.json/.csv/.parquet/.txt file of records")
    p.add_argument("--text-field", default="text", help="column name for the text to analyze (default: 'text')")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--workload", required=True, help="query file (.txt one per line / .json list) or a literal query")
    p.add_argument("--description", required=True, help="dataset description (literal string or file path)")
    p.add_argument("--attributes", default=None, help="JSON file with attributes (skips workload analysis)")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--epsilon", type=float, default=1.0, help="global privacy budget (finite, > 0)")
    p.add_argument("--delta", type=float, default=None, help="default: 1/(n ln n)")
    p.add_argument("--budget-mode", choices=("tight", "loose"), default=SHIPConfig.budget_mode,
                   help="tight: Theorem 1 conversion (default); loose: Appendix B.1 delta/3 + 2delta/3 split")
    p.add_argument("--subsample-rate", type=float, default=None)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--tau-close", type=float, default=10.0)

    p.add_argument("--no-dedup-open-before-comp", action="store_false", dest="dedup_open_before_comp",
                   help="skip de-duplication of C_open before the complementary proposal")
    p.add_argument("--dedup-after-close", action="store_true",
                   help="also de-duplicate C_surv after closed thresholding (Algorithm 3)")
    p.add_argument("--no-normalize-none", action="store_false", dest="normalize_none_variants",
                   help="skip the level-1 LLM mapping of null-like categories to none")
    p.add_argument("--no-disambiguate", action="store_false", dest="disambiguate_tree_names",
                   help="skip whole-tree disambiguation of repeated category names")
    p.add_argument("--no-auto-example-path", action="store_false", dest="auto_example_path",
                   help="do not generate granularity example paths for attributes without one")
    p.add_argument("--max-text-chars", type=int, default=20000)

    p.add_argument("--attribute-model", default=SHIPConfig.attribute_model)
    p.add_argument("--taxonomy-model", default=SHIPConfig.taxonomy_model)
    p.add_argument("--classification-model", default=SHIPConfig.classification_model)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--thinking-budget", type=int, default=None)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--cache-path", default=".cache/llm_cache.sqlite", help="'' disables caching")
    p.add_argument("--vertexai", action="store_true")

    p.add_argument("--aim-max-model-size", type=float, default=80.0)
    p.add_argument("--aim-max-iters", type=int, default=1000)
    p.add_argument("--num-synthetic-rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for subsampling and DP noise (default: 42)")
    p.add_argument("--save-private-allocation", action="store_true",
                   help="also save the NON-private allocation table (debugging / analysis only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = SHIPConfig(
        epsilon=args.epsilon, delta=args.delta, subsample_rate=args.subsample_rate, depth=args.depth,
        budget_mode=args.budget_mode,
        tau_close=args.tau_close, dedup_open_before_comp=args.dedup_open_before_comp,
        dedup_after_close=args.dedup_after_close, normalize_none_variants=args.normalize_none_variants,
        disambiguate_tree_names=args.disambiguate_tree_names, auto_example_path=args.auto_example_path,
        max_text_chars=args.max_text_chars, attribute_model=args.attribute_model,
        taxonomy_model=args.taxonomy_model, classification_model=args.classification_model,
        temperature=args.temperature, thinking_budget=args.thinking_budget, max_workers=args.max_workers,
        cache_path=args.cache_path or None, vertexai=args.vertexai,
        aim_max_model_size=args.aim_max_model_size, aim_max_iters=args.aim_max_iters,
        num_synthetic_rows=args.num_synthetic_rows, seed=args.seed,
        save_private_allocation=args.save_private_allocation,
    )
    texts = load_records(args.data, args.text_field, args.max_records)
    workload = load_workload(args.workload)
    description = read_text_or_literal(args.description)
    # optionally load attributes directly. If None, SHIP will determine them from the workload.
    attributes = load_attributes(args.attributes) if args.attributes else None

    result = SHIP(cfg).run(texts, workload, description, attributes)
    result.save(args.output_dir, save_private=cfg.save_private_allocation,
                export_eval_compat=cfg.export_eval_compat)
    logging.info("Saved SHIP release to %s (usage: %s)", args.output_dir, result.usage)


if __name__ == "__main__":
    main()
