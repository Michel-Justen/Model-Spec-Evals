# Results

Independent replication of OpenAI's Model Spec compliance eval, using OpenAI's own
[harness](https://github.com/openai/model_spec_evals) and
[dataset](https://github.com/openai/model_spec_dataset) (587 runnable prompts, unmodified), extended
to two models OpenAI has not reported: GPT-5.5 Thinking and GPT-5.6 Sol. Settings, validation, and
caveats are in [methodology.md](methodology.md). Everything below is reproducible (see Reproduce); the
raw per-answer logs are on Google Drive (see the README).

This page reports the raw numbers and comparisons and keeps interpretation to a minimum.

## Overall compliance

| Model | Compliance | 95% CI | Samples/prompt | OpenAI published |
|---|---|---|---|---|
| GPT-5 Thinking (Aug 2025) | 90.0% | ±0.6 | 5 | 89% |
| GPT-5.4 Thinking (Mar 2026) | 85.5% | ±0.6 | 5 | 87% |
| GPT-5.5 Thinking (Apr 2026) | 85.7% | ±0.5 | 5 | not reported |
| GPT-5.6 Sol (Jul 2026) | 85.6% | ±0.4 | 10 | not reported |

Every model: `reasoning_effort=high`, `gpt-5` grader, 5 gradings per answer, 587 prompts. CIs are
fixed-benchmark (sampling-noise) 95% intervals. Machine-readable: [results/summary.csv](results/summary.csv).

Our GPT-5 Thinking and GPT-5.4 numbers differ from OpenAI's published figures by +1.0 and −1.5 points
respectively. We did not fully explain the −1.5 GPT-5.4 gap; the leading candidates are an undisclosed
higher reasoning effort (we ran at `high`, which is not the maximum) and/or an internal checkpoint
differing from the public one. See Validation in [methodology.md](methodology.md).

## Between-model comparisons (paired)

Every model was run on the same 587 prompts, so we compare with a paired test on per-prompt
compliance (`src/compare.py`):

| Comparison | Difference | p |
|---|---|---|
| GPT-5 Thinking vs GPT-5.4 | 4.5 pts | < 0.0001 |
| GPT-5 Thinking vs GPT-5.5 | 4.3 pts | 0.0001 |
| GPT-5 Thinking vs GPT-5.6 Sol | 4.3 pts | 0.0001 |
| GPT-5.5 vs GPT-5.6 Sol | 0.1 pts | 0.92 (not significant) |

GPT-5.4, GPT-5.5, and GPT-5.6 Sol are each lower than GPT-5 Thinking, and not statistically
distinguishable from one another.

## By top-level section (GPT-5 Thinking vs GPT-5.6 Sol)

| Section | GPT-5 Thinking | GPT-5.6 Sol | Difference | Prompts |
|---|---|---|---|---|
| chain_of_command | 93.1% | 86.9% | −6.2 | 96 |
| seek_truth | 91.5% | 85.2% | −6.2 | 120 |
| best_work | 86.9% | 81.8% | −5.1 | 84 |
| stay_in_bounds | 87.2% | 83.8% | −3.4 | 149 |
| style | 91.3% | 89.4% | −1.9 | 138 |

(Per-section figures for GPT-5.4 and GPT-5.5 are reproducible from the logs with `src/drilldown.py`.)

## Largest sub-section differences (GPT-5 Thinking vs GPT-5.6 Sol)

Small-n rows are noisy; the prompt count (n) is shown for each.

| Sub-section | Difference | n |
|---|---|---|
| transformation_exception | −32.2 | 9 |
| assume_best_intentions | −19.2 | 12 |
| ask_clarifying_questions | −19.0 | 20 |
| support_programmatic_use | −12.5 | 24 |
| be_thorough_but_efficient | −10.7 | 28 |
| express_uncertainty | −10.0 | 36 |

## Prompt-level agreement (GPT-5 Thinking vs GPT-5.6 Sol)

Each prompt binarized to pass/fail (majority of its samples):

| | Prompts |
|---|---|
| Both pass | 490 |
| Both fail | 30 |
| Sol fails, GPT-5 Thinking passes | 46 |
| GPT-5 Thinking fails, Sol passes | 21 |

An exploratory classification of Sol's 46 losses by an LLM (`gpt-4o-mini`, **not hand-verified**;
per-prompt labels in [results/failure_directions.json](results/failure_directions.json)): 23 labeled
"too permissive," 15 "over-refusal," 8 "other (accuracy / format / honesty)." Treat these labels as
indicative only.

## Reproduce

```bash
./setup.sh
export OPENAI_API_KEY=...
python src/run_eval.py --model openai/gpt-5.6-sol --preset moderate \
    --epochs 5 --grader-samples 5 --reasoning-effort high --log-dir runs/sol
python src/analyze.py runs/sol/*.eval --pool
```

Re-run the analyses on our exact outputs (download the Drive logs into `runs/` first, one folder per
model):

```bash
python src/compare.py --a runs/anchor-gpt5-thinking --b runs/sol \
    --labels "GPT-5 Thinking" "GPT-5.6 Sol"
python src/drilldown.py
python src/failure_direction.py
```

Model ids, settings, and validation are in [methodology.md](methodology.md).

## Compute

~$4,900 of OpenAI API credits (the grader dominates cost), funded by a BlueDot Rapid AI Safety Grant.
Per-run token counts are in the raw logs.
