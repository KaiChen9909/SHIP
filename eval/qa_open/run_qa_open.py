"""CLI entry point for Open-Ended Symmetric QA evaluation (`python -m eval.qa_open.run_qa_open`)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from dotenv import load_dotenv

from eval.common.code_agent import DEFAULT_EVAL_MODEL

from .evaluator import SymmetricEvaluator


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="eval.qa_open.run_qa_open", description="Symmetric Open-Ended QA Evaluation.")
    p.add_argument("--target-dir", required=True, help="Target SHIP output directory.")
    p.add_argument("--gt-dir", required=True, help="Ground-truth hierarchical directory.")
    p.add_argument("--workload", required=True, help="Path to workload JSON file.")
    p.add_argument("--dataset-name", default="wildchat")
    p.add_argument("--dataset-description", default="")
    p.add_argument("--target-algo", default="ship")
    p.add_argument("--model", default=DEFAULT_EVAL_MODEL)
    p.add_argument("--vertexai", action="store_true")
    p.add_argument("--num-rounds", type=int, default=5)
    p.add_argument("--k", type=int, default=None, help="Max atomic statements per report (default: exhaustive).")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--parallel-num", type=int, default=5)
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--overwrite-gt-cache", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluator = SymmetricEvaluator(model_name=args.model, vertexai=args.vertexai)
    report = asyncio.run(
        evaluator.evaluate(
            target_data_dir=args.target_dir,
            workload_file_path=args.workload,
            gt_data_dir=args.gt_dir,
            target_algo_name=args.target_algo,
            dataset_name=args.dataset_name,
            dataset_description=args.dataset_description,
            num_rounds=args.num_rounds,
            k=args.k,
            validation_batch_size=args.batch_size,
            parallel_num=args.parallel_num,
            output_report_path=args.output,
            overwrite_cache=args.overwrite_cache,
            overwrite_gt_cache=args.overwrite_gt_cache,
        )
    )
    print(json.dumps(report["multi_round_stats"], indent=2))


if __name__ == "__main__":
    main()
