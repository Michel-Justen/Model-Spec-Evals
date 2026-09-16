# Results

Independent replication of OpenAI's Model Spec compliance eval, using OpenAI's own
[harness](https://github.com/openai/model_spec_evals) and
[dataset](https://github.com/openai/model_spec_dataset) (587 runnable prompts, unmodified), extended
to three models OpenAI has not reported: GPT-5.5 Thinking, GPT-5.6 Sol and GPT-6 Astra. Settings,
validation, and caveats are in [methodology.md](methodology.md). Everything below is reproducible (see
Reproduce); the raw per-answer logs are on Google Drive (see the README).

This page reports the raw numbers and comparisons and keeps interpretation to a minimum.

## Overall compliance

| Model | Compliance | 95% CI | Samples/prompt | OpenAI published |
|---|---|---|---|---|
| GPT-5 Thinking (Aug 2025) | 90.0% | ±0.6 | 5 | 89% |
| GPT-5.4 Thinking (Mar 2026) | 85.5% | ±0.7 | 5 | 87% |
| GPT-5.5 Thinking (Apr 2026) | 85.7% | ±0.6 | 5 | not reported |
| GPT-5.6 Sol (Jul 2026) | 85.6% | ±0.4 | 10 | not reported |
| GPT-6 Astra (Sep 2026) | 89.2% | ±0.6 | 5 | not reported |

Every model: `reasoning_effort=high`, `gpt-5` grader, 5 gradings per answer, 587 prompts. CIs are
fixed-benchmark (sampling-noise) 95% intervals, computed from the Bessel-corrected per-prompt variance.
Machine-readable: [results/summary.csv](results/summary.csv). Full output:
[results/astra_final_analysis_5ep.txt](results/astra_final_analysis_5ep.txt).

Our GPT-5 Thinking and GPT-5.4 numbers differ from OpenAI's published figures by +1.0 and −1.5 points
respectively. We did not fully explain the −1.5 GPT-5.4 gap; the leading candidates are an undisclosed
higher reasoning effort (we ran at `high`, which is not the maximum) and/or an internal checkpoint
differing from the public one. See Validation in [methodology.md](methodology.md).

### API filter blocks

A small share of answers are not the model's own: OpenAI's API refuses some prompts with a fixed
message ("This content was flagged for possible cybersecurity risk…"), which the harness then grades.
The table above scores them as returned, which is the harness default. Excluding them changes little
and reorders nothing:

| Model | As published | Filter-blocked answers excluded | Blocked answers / prompts |
|---|---|---|---|
| GPT-5 Thinking | 89.98% | 90.10% | 15 / 3 |
| GPT-5.4 Thinking | 85.45% | 85.57% | 10 / 2 |
| GPT-5.5 Thinking | 85.72% | 86.34% | 55 / 14 |
| GPT-5.6 Sol | 85.64% | 86.17% | 78 / 9 |
| GPT-6 Astra | 89.18% | 89.58% | 30 / 6 |

A second filter (biosecurity) returns an API error rather than text; those answers are never scored,
for any model. Three prompts were blocked this way in every Astra run, so Astra's figures are computed
over 584 of the 587 prompts.

## Between-model comparisons (paired)

Every model was run on the same prompts, so we compare with a paired test on per-prompt compliance
(`src/analyze_scores.py`; the older `src/compare.py` does the same for a single pair):

| Comparison | Difference | 95% CI | p |
|---|---|---|---|
| GPT-5 Thinking vs GPT-5.4 | 4.5 pts | ±1.9 | 0.0000034 |
| GPT-5 Thinking vs GPT-5.5 | 4.3 pts | ±2.1 | 0.000077 |
| GPT-5 Thinking vs GPT-5.6 Sol | 4.3 pts | ±2.2 | 0.000098 |
| GPT-5 Thinking vs GPT-6 Astra | 0.9 pts | ±2.2 | 0.41 (not significant) |
| GPT-6 Astra vs GPT-5.4 | 3.6 pts | ±2.1 | 0.00068 |
| GPT-6 Astra vs GPT-5.5 | 3.4 pts | ±2.0 | 0.00073 |
| GPT-6 Astra vs GPT-5.6 Sol | 3.4 pts | ±1.8 | 0.00013 |
| GPT-5.5 vs GPT-5.6 Sol | 0.1 pts | — | 0.92 (not significant) |

GPT-5.4, GPT-5.5, and GPT-5.6 Sol are each lower than GPT-5 Thinking and not statistically
distinguishable from one another. GPT-6 Astra is higher than all three of them, and **not**
distinguishable from GPT-5 Thinking.

## By top-level section

Prompt-weighted compliance, scored as published:

| Section | Prompts | GPT-5 Thinking | GPT-5.4 | GPT-5.5 | GPT-5.6 Sol | GPT-6 Astra |
|---|---|---|---|---|---|---|
| chain_of_command | 96 | 93.1% | 88.5% | 87.9% | 86.9% | 89.2% |
| seek_truth | 120 | 91.5% | 86.2% | 86.5% | 85.2% | 92.2% |
| best_work | 84 | 86.9% | 81.9% | 81.7% | 81.8% | 88.8% |
| stay_in_bounds | 149 | 87.2% | 83.4% | 83.4% | 83.8% | 84.5% |
| style | 138 | 91.3% | 87.1% | 88.6% | 89.4% | 91.8% |

Sol's largest drop against GPT-5 Thinking is seek_truth (−6.3) and chain_of_command (−6.2). Astra
recovers most of that ground but not on chain_of_command (−3.9 vs GPT-5 Thinking). Full output:
[results/section_scores_5ep.txt](results/section_scores_5ep.txt).

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

## GPT-6 Astra

Astra was run in two batches on the same settings (3 epochs on 2026-09-11, 2 more on 2026-09-15),
pooled per prompt: 2,912 scored answers over 584 prompts. Requests were verified from the logged API
payloads (`src/verify_requests.py`): Responses API, `reasoning.effort=high`, `summary=detailed`, and a
grader input identical in shape to the other models'. Total API cost ≈ $954.

**No drift between the batches.** The 2-epoch batch scored 89.28% and the 3-epoch batch 89.21%; paired
per prompt, the difference is −0.09 points (95% CI −1.27 to +1.09, p=0.89). There is no sign the served
model changed between runs, so pooling is justified.

### Evaluation awareness (does Astra know it is being tested?)

An AI judge (`gpt-5.4`, the model OpenAI uses to monitor Astra) read every answer plus the reasoning
summary the API returned, and flagged five kinds of evaluation awareness with a verbatim quote that is
re-checked in code (`src/eval_awareness_judge.py`). Validation rows were mixed blind into the queue:
**30 of 30 planted eval-aware sentences were caught, and 0 of 10 decoys** (ordinary uses of "test",
"benchmark", "rubric") were wrongly flagged, in both runs.

**Result: 0 of 2,876 judged answers showed evaluation awareness** at medium-or-higher confidence
(95% CI 0–0.13%). Only **1,877 of those answers had a reasoning summary** (65%); the rest were judged
on the final answer alone. Restricted to answers with a summary: 0 of 1,877 (95% CI 0–0.20%). One
epochs-1–3 flag was ambiguous on manual review, and every low-confidence flag was a false positive
(Astra correctly identifying a prompt injection inside pasted data).

**This is a weak instrument and does not show Astra is unaware.** OpenAI never returns raw chain of
thought; the summaries are paraphrases written by a separate model, a third of answers have none (over
half in the safety section), and a same-family judge cannot detect unverbalized awareness. OpenAI's own
system card, which reads raw reasoning, reports Astra as *more* evaluation-aware than GPT-5.6 Sol
(9.6% vs 2.8% of trajectories in deployment simulation), and Apollo Research measured 41% at high
effort and 51% at maximum. Method and limitations:
[docs/EVAL_AWARENESS_SEARCH_METHODS.md](docs/EVAL_AWARENESS_SEARCH_METHODS.md).

### Contamination (are the eval's prompts and rubrics in Astra's training data?)

The dataset and its grading rubrics were published on 2026-03-24, about five weeks before Astra's
2026-04-30 knowledge cutoff, and carry no canary string. Two probes (`src/contamination_probes.py`)
compared Astra against GPT-5.6 Sol and GPT-5, whose cutoffs precede the dataset's release:

- **Rubric reconstruction.** Shown a prompt and three example rubrics, Astra reproduced 0.6% of the
  real rubric's distinctive five-word phrases — the same as GPT-5 (0.6%), and 0.0% against a
  deliberately mismatched rubric.
- **Verbatim completion.** Given the first half of a dataset prompt, Astra recovered 22.1% of the rest
  (ROUGE-L), against 17.5% for GPT-5 — but it also scored 20.4% on freshly written control prompts that
  cannot be in any training data (GPT-5: 18.0%). Within Astra, real minus control is +1.8 points
  (95% CI −2.5 to +6.8): no evidence of memorization.

**Verdict: null, and the probes are weak.** They catch only near-verbatim recall: of 15 Model Spec
passages — public since 2024 and almost certainly in training — Astra reproduced just one
near-word-for-word (a 32-word exact run) and recalled the other 14 no better than post-cutoff text.
Full numbers: [results/contamination_report.md](results/contamination_report.md); pre-registered
decision rules and manual review: [results/contamination_verdict.md](results/contamination_verdict.md).

## Reproduce

```bash
./setup.sh
export OPENAI_API_KEY=...
python src/run_eval.py --model openai/gpt-5.6-sol --preset moderate \
    --epochs 5 --grader-samples 5 --reasoning-effort high --log-dir runs/sol
python src/analyze.py runs/sol/*.eval --pool
```

GPT-6 Astra needs inspect-ai ≥ 0.3.263 (earlier versions detect reasoning models by the string
`gpt-5`, so Astra silently ran through the Chat Completions API at its default effort) and
`--reasoning-summary detailed` to retain reasoning summaries:

```bash
python src/run_eval.py --model openai/gpt-6-astra --preset moderate \
    --epochs 5 --grader-samples 5 --reasoning-effort high \
    --reasoning-summary detailed --log-dir runs/astra
python src/verify_requests.py runs/astra/*.eval --candidate openai/gpt-6-astra \
    --expect-effort high --expect-summary detailed
```

Re-run the analyses on our exact outputs (download the Drive logs into `runs/` first, one folder per
model):

```bash
python src/analyze_scores.py | tee results/astra_final_analysis_5ep.txt
python src/section_scores.py | tee results/section_scores_5ep.txt
python src/compare.py --a runs/anchor-gpt5-thinking --b runs/sol \
    --labels "GPT-5 Thinking" "GPT-5.6 Sol"
python src/drilldown.py
python src/failure_direction.py
```

Model ids, settings, and validation are in [methodology.md](methodology.md).

## Compute

~$4,900 of OpenAI API credits for GPT-5 Thinking, GPT-5.4, GPT-5.5 and GPT-5.6 Sol (the grader
dominates cost), funded by a BlueDot Rapid AI Safety Grant. The GPT-6 Astra run (≈$954) and the
contamination and evaluation-awareness probes (≈$20) were funded by a reader donation. Per-run token
counts are in the raw logs.
