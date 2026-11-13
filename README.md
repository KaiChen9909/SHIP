# SHIP: Structured and Hierarchical Insights with Privacy

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

## Dataset

All datasets investigated in the paper are open-sourced. We provided the link to download them. Therefore, we do not include the raw data here.

## Run

```bash
python -m ship.run \
  --data data/records.jsonl --text-field text \
  --workload workloads/wildchat/topic_workload.json \
  --description "Conversations between users and an AI assistant" \
  --epsilon 1.0 --output-dir outputs/run_eps1 \
  [--attributes workloads/wildchat/topic_attributes.json] [--subsample-rate 1.0] [--depth 3]
```

Optional steps are on by default and can be switched off with `--no-dedup-open-before-comp`,
`--no-normalize-none`, `--no-disambiguate`, `--no-auto-example-path`; `--dedup-after-close`
additionally de-duplicates after closed thresholding (Algorithm 3 line 25).

`--workload` accepts a .txt file (one query per line), a JSON list, or the benchmark format
`{"1": {"title": ..., "description": ...}}`. `--attributes` skips workload analysis; it accepts a
JSON list of `{"name", "description"}` or `{"domain_annotations": {name: {...}}}`.
Attribute inputs do not need hierarchy examples. Whether attributes come from a file
or workload analysis, the taxonomy model generates a missing `example_path` for each
attribute in a separate step by default, using the public dataset description and
attribute description. Each example is generated one level per LLM call: the model
receives the total depth, current level, and previously generated parent path, and
returns only the current category. A depth-d example normally requires d sequential
calls per attribute. These examples are inserted into the taxonomy prompts and
saved in the output `attributes.json`. `--no-auto-example-path` disables this step.

`--budget-mode tight` (default) uses Theorem 1's exact conversion parameter
`delta_conversion = (delta - delta_tax) / (1 - delta_tax)`.
`--budget-mode loose` follows Appendix B.1: `delta_tax = delta/3` and
`delta_conversion = 2*delta/3`. Both modes split the resulting `rho_total` equally
between taxonomy construction and AIM by default, and both meet the target
`(epsilon, delta)` budget under the paper's assumptions. The tight mode permits
a slightly larger `rho_total` and therefore slightly less noise. With subsampling,
the selected mode applies to the amplified-back run-level budget.
`privacy.json` records `budget_mode` and `delta_conversion`.

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
