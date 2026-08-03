# Model Spec compliance — independent replication

An independent measurement of how well OpenAI's recent models follow OpenAI's own
[Model Spec](https://github.com/openai/model_spec), using OpenAI's public eval
[harness](https://github.com/openai/model_spec_evals) and 587-prompt
[dataset](https://github.com/openai/model_spec_dataset) (both public domain), extended to the
recent flagships OpenAI has **not** reported scores for.

**Summary:** in this replication, GPT-5 Thinking scores highest; GPT-5.4, GPT-5.5, and GPT-5.6 Sol
all sit near 85–86%. The two models OpenAI has not reported — **GPT-5.5 Thinking** and **GPT-5.6
Sol** — score below GPT-5 Thinking on OpenAI's own spec (paired test, p ≈ 0.0001) and are
statistically indistinguishable from GPT-5.4 and from each other. Absolute numbers reproduce
OpenAI's published figures to within ~1.5 points (see [methodology.md](methodology.md)).

| Model | Compliance (ours) | OpenAI published |
|---|---|---|
| GPT-5 Thinking | 90.0% | 89% |
| GPT-5.4 Thinking | 85.5% | 87% |
| GPT-5.5 Thinking | 85.7% | *unreported* |
| GPT-5.6 Sol | 85.6% | *unreported* |

👉 **Full results and comparisons: [RESULTS.md](RESULTS.md).**
**Method, settings, and validation: [methodology.md](methodology.md).**

## Repo map

- **[RESULTS.md](RESULTS.md)** — the raw numbers: overall scores, paired comparisons, and
  per-section / sub-section breakdowns, plus reproduce steps
- **[methodology.md](methodology.md)** — exact settings, grader, sampling, and validation reasoning
- `results/` — `summary.csv` and `failure_directions.json` (per-prompt audit labels)
- `src/` — the code:
  - `run_eval.py` — run the eval (robustness flags: retry, fail-on-error, timeout, connections)
  - `analyze.py` — logs → table + CSV + chart, with pooled clustered CIs (`--pool`)
  - `compare.py` — **paired** significance test between two models on the shared prompts
  - `make_chart.py` — the compliance charts (`--ours-only` and the OpenAI-context version)
  - `drilldown.py` — where a model loses ground, by spec sub-section, with examples
  - `failure_direction.py` — classifies failures as over- vs under-cautious
  - `cost_model.py` — estimate $ before a run · `check_models.py` — pre-flight model reachability
- `config/` — model ids, grader, presets, token prices
- `setup.sh` — one command: Python env, clone OpenAI's harness + dataset, validate

## Raw logs (full transcripts)

The raw `.eval` logs (~2 GB — every prompt, model response, and grader rationale) are **not in git**;
they're shared via Google Drive so the analysis layer can be re-run on our exact outputs without
re-spending on the eval:

> **Drive folder:** https://drive.google.com/drive/folders/12YmVY3f3-nq94FBv2jgeYa6BT0W_PtvB — one
> folder per model; download into `runs/`, then run the analysis scripts (see
> [RESULTS.md](RESULTS.md#reproduce)).

This mirrors OpenAI's own posture (public harness + dataset + scores; transcripts not published).
The dataset prompts themselves are public (CC0) in OpenAI's repo, keyed by ID.

## Reproduce

```bash
./setup.sh
export OPENAI_API_KEY=...
python src/check_models.py
python src/run_eval.py --model openai/gpt-5.6-sol --preset moderate \
    --epochs 5 --grader-samples 5 --reasoning-effort high --log-dir runs/sol
python src/analyze.py runs/sol/*.eval --pool
```

Compute: ~$4,900 of API credits, funded by a BlueDot Rapid AI Safety Grant.
