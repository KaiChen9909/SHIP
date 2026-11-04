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
  prompts/                Jinja2 prompt templates (Appendix G, Figs. 5-8)
    propose_categories.txt       Fig. 5  open discovery
    reclassify.txt               Fig. 6  re-classification
    complementary_categories.txt Fig. 7  complementary proposal (output-format line added)
    deduplicate_categories.txt   Fig. 8  de-duplication
    attribute_determination.txt  (not given in the paper; empty, fill in before use)
utils/
  llm.py                  Gemini API client (retries, sqlite cache, JSON parsing, thread pool)
  dp.py                   zCDP<->DP, Gaussian thresholding, budget split, Poisson subsampling
  aim.py                  AIM synthesizer parameterised by rho (adapted from private-pgm)
  io.py                   data / workload loading, prompt rendering
eval/                     (to be added)
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
  --workload workload.txt \
  --description "Conversations between users and an AI assistant" \
  --epsilon 1.0 --output-dir outputs/run_eps1 \
  [--attributes attributes.json] [--subsample-rate 0.02] [--depth 3]
```

`--attributes` skips workload analysis; the file is a JSON list of
`{"name", "description", "example_path"?, "reference_categories"?}`.
Otherwise `ship/prompts/attribute_determination.txt` must be filled in first
(see the docstring of `ship/attributes.py` for its variables and expected output).

Outputs (`--output-dir`): `taxonomy.json` (DP taxonomy with noisy counts),
`synthetic.csv` (DP synthetic table with `<attr> (level k)` / `<attr> (path)` columns),
`attributes.json`, `privacy.json`, `usage.json`.
`--save-private-allocation` additionally writes the **non-private** record allocation
to `private_DO_NOT_RELEASE/` for analysis only.
