# SHIP: Structured and Hierarchical Insight with Privacy

This repository implements the SHIP algorithm from `paper/paper.pdf`.

```
ship/                     main algorithm
  pipeline.py             Algorithm 2 (end-to-end SHIP + privacy accounting)
  attributes.py           workload analysis  A <- M_attr(W)
  taxonomy.py             Algorithm 1/3 (hierarchical hybrid DP taxonomy + allocation)
  synthesis.py            path encoding + AIM + per-level decoding
  config.py               SHIPConfig / Attribute
  run.py                  CLI
  prompts/                prompt templates (plain variable substitution, no control flow)
    attribute_determination.txt  workload -> attributes
    sample_path.txt              public granularity example path per attribute
    propose_categories.txt       Fig. 5  open discovery
    identify_none_categories.txt level-1 null-like category detection
    deduplicate_categories.txt   Fig. 8  de-duplication (list of {raw, canonical})
    complementary_categories.txt Fig. 7  complementary proposal
    reclassify.txt               Fig. 6  re-classification
    disambiguate_tree_names.txt  whole-tree renaming of repeated names
workloads/                benchmark workloads + pre-extracted attributes (chatdoctor, wildchat, cfpb)
utils/
  llm.py                  Gemini API client (retries, sqlite cache, JSON parsing, thread pool)
  dp.py                   zCDP<->DP, Gaussian thresholding, budget split, Poisson subsampling
  aim.py                  AIM synthesizer parameterised by rho (adapted from private-pgm)
  io.py                   data / workload loading, prompt rendering
eval/
  common/                 sandboxed Python-REPL code agent (google-genai tool calling),
                          taxonomy/table helpers (tree_inspect, tabular_stat, tabular_crosstab)
  qa/                     closed-ended QA: GT question bank -> answer on target -> 1-5 rubric (Acc@T4)
  qa_open/                open-ended QA: reports -> atomic statements -> cross verification (P/R/F1@T4)
```

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=...        # or put it in .env; use --vertexai for Vertex AI
```

## Run

```bash
python -m ship.run \
  --data data/records.jsonl --text-field text \
  --workload workloads/wildchat/topic_workload.json \
  --description "Conversations between users and an AI assistant" \
  --epsilon 1.0 --output-dir outputs/run_eps1 \
  [--attributes workloads/wildchat/topic_attributes.json] [--subsample-rate 0.02] [--depth 3]
```

Optional steps are on by default and can be switched off with `--no-dedup-open-before-comp`,
`--no-normalize-none`, `--no-disambiguate`, `--no-auto-example-path`; `--dedup-after-close`
additionally de-duplicates after closed thresholding (Algorithm 3 line 25).

`--workload` accepts a .txt file (one query per line), a JSON list, or the benchmark format
`{"1": {"title": ..., "description": ...}}`. `--attributes` skips workload analysis; it accepts a
JSON list of `{"name", "description", "example_path"?}` or `{"domain_annotations": {name: {...}}}`.
Attributes without `example_path` get one generated from public information.

Outputs (`--output-dir`): `taxonomy.json` (DP taxonomy with noisy counts),
`category_tree.json` (same tree in the schema used by `eval/`),
`synthetic.csv` (DP synthetic table with `<attr> (level k)` / `<attr> (path)` columns),
`attributes.json`, `privacy.json`, `usage.json`.
`--save-private-allocation` additionally writes the **non-private** record allocation
to `private_DO_NOT_RELEASE/` for analysis only.

## Evaluation

Both protocols compare a released insight (`--target-dir`, a SHIP output directory) with a
ground-truth directory (`--gt-dir`). The ground truth is produced outside this repository (the paper
uses a separate non-DP summarization, or labels known by construction for synthetic corpora); it
must hold `category_tree.json` and a per-record table: either `synthetic.csv` in the SHIP layout
(`<attr> (level k)` / `<attr> (path)` columns) or `allocated_paths.csv` with one `a > b > c` path
column per attribute (level columns are derived automatically).

```bash
# closed-ended QA (default 2 x 3 x 50 = 300 questions, generated once on the GT and cached;
# every answer is scored 1-5 by an LLM judge using the paper's rubric for its question type)
python -m eval.qa.run_qa --target-dir outputs/run_eps1 --gt-dir outputs/wildchat_gt \
  --workload workloads/wildchat/topic_workload.json --dataset-name wildchat

# open-ended QA (5 rounds, Recall / Precision / F1 at score >= 4)
python -m eval.qa_open.run_qa_open --target-dir outputs/run_eps1 --gt-dir outputs/wildchat_gt \
  --workload workloads/wildchat/topic_workload.json --dataset-name wildchat --num-rounds 5
```

The evaluation agent defaults to `gemini-3.5-flash-lite` (`--model` to change). Agent code can
only read the released files of the directory it evaluates (`synthetic.csv`, `taxonomy.json`,
`category_tree.json`, ...); `private_DO_NOT_RELEASE/` is blocked. The sandbox is a guard for a
cooperative model, not a security boundary. Reports are written to the target directory unless
`--output` is given.
