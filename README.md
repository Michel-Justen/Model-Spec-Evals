# Model Spec adherence

Scripts to run OpenAI's Model Spec eval against models OpenAI hasn't published adherence
scores for — GPT-5.5 and the GPT-5.6 family (Sol, Terra, Luna).

OpenAI open-sourced the eval harness and the 596-prompt dataset, but only reported scores
through GPT-5.4 Thinking. This wraps their harness so you can point it at newer models,
estimate what a run will cost before spending anything, and turn the logs into a table and
a chart.

## Layout

- `setup.sh` — makes a Python env, clones OpenAI's harness + dataset, checks the dataset loads
- `config/models.yaml` — which models to test, the grader, and the run presets
- `config/prices.yaml` — token prices used by the cost estimate
- `src/run_eval.py` — run the eval for a given model and preset
- `src/cost_model.py` — estimate the dollar cost of a run before you start
- `src/check_models.py` — confirm the model ids are reachable before spending
- `src/analyze.py` — turn the `.eval` logs into a table, a CSV, and a bar chart
- `methodology.md` — how the eval works and which settings match OpenAI's
- `Adherence.png` — chart from an early 50-prompt pilot (illustrative, not a final result)

## Running it

```
./setup.sh
export OPENAI_API_KEY=...
python src/check_models.py
python src/run_eval.py --model openai/gpt-5.6-sol --preset moderate
python src/analyze.py --logs-dir vendor/model_spec_evals/logs
```

The grader re-reads the entire Model Spec on every call, so it drives almost all of the cost.
Run `python src/cost_model.py` to see the breakdown before starting a big run.

Built on OpenAI's [model_spec_evals](https://github.com/openai/model_spec_evals) and
[model_spec_dataset](https://github.com/openai/model_spec_dataset), both public domain.
