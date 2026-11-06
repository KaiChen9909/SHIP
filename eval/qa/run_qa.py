"""CLI entry point for Closed-Ended Direct QA evaluation (`python -m eval.qa.run_qa`)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from dotenv import load_dotenv

from eval.common.code_agent import DEFAULT_EVAL_MODEL

from .evaluator import QueryAnswerEvaluator


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="eval.qa.run_qa", description="Closed-Ended Direct QA Evaluation for SHIP.")
    p.add_argument("--target-dir", required=True, help="Target SHIP output directory (containing synthetic.csv).")
    p.add_argument("--gt-dir", required=True, help="Ground-truth hierarchical directory.")
    p.add_argument("--workload", required=True, help="Path to workload JSON file.")
    p.add_argument("--dataset-name", default="wildchat")
    p.add_argument("--dataset-description", default="")
    p.add_argument("--target-algo", default="ship")
    p.add_argument("--model", default=DEFAULT_EVAL_MODEL)
    p.add_argument("--vertexai", action="store_true")
    p.add_argument("--questions-per-type", type=int, default=50)
    p.add_argument("--num-rounds", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--parallel-num", type=int, default=10)
    p.add_argument("--question-bank", default=None, help="Optional pre-generated *_question_bank.json.")
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluator = QueryAnswerEvaluator(model_name=args.model, vertexai=args.vertexai)
    report = asyncio.run(
        evaluator.evaluate_async(
            target_data_dir=args.target_dir,
            workload_file_path=args.workload,
            gt_data_dir=args.gt_dir,
            target_algo_name=args.target_algo,
            dataset_name=args.dataset_name,
            dataset_description=args.dataset_description,
            num_rounds=args.num_rounds,
            questions_per_type=args.questions_per_type,
            batch_size=args.batch_size,
            parallel_num=args.parallel_num,
            question_bank_path=args.question_bank,
            output_report_path=args.output,
            overwrite_cache=args.overwrite_cache,
            max_questions=args.max_questions,
        )
    )
    print(
        json.dumps(
            {
                "primary_accuracy_t4": report["primary_accuracy_t4"],
                "overall_average_score": report["overall_average_score"],
                "overall_mae": report["overall_mae"],
                "breakdown_by_question_type": report["breakdown_by_question_type"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
