# Methodology

How this replication measures Model Spec compliance, the exact settings, and how we validated it.
Results are in [RESULTS.md](RESULTS.md).

## What we measure

Compliance = fraction of prompts where the candidate model's response is judged **compliant** with
the relevant clause of the [OpenAI Model Spec](https://github.com/openai/model_spec). We use OpenAI's
own open harness ([`model_spec_evals`](https://github.com/openai/model_spec_evals), built on Inspect
AI) and dataset ([`model_spec_dataset`](https://github.com/openai/model_spec_dataset), CC0),
**unmodified**, so the pipeline is identical to theirs.

## The pipeline (verified from source)

1. **Prompts.** 596 authored prompts across 225 focus areas and 5 top-level spec sections
   (`style`, `chain_of_command`, `best_work`, `seek_truth`, `stay_in_bounds`). 9 are marked `skip`
   → **587 runnable**. Confirmed locally: 587 samples load; section counts match OpenAI's.
2. **Candidate.** Each prompt is a short system/developer/user conversation; the candidate model
   produces the final assistant turn. Sampled `epochs` times per prompt.
3. **Grader.** For each candidate response, the grader (`gpt-5`) receives grading template + **the
   full Model Spec** (pinned to the 2025-12-18 release, commit `8d2405d…`) + the prompt's rubric.
   It returns a structured critique with a 1–7 compliance score. Sampled `num_grader_samples` times;
   the **median** is taken.
4. **Binarize.** compliance ≥ 6 → compliant (1.0), else 0.0. Metric = mean per `top_level_section`,
   plus `all` over everything.

## Settings we used

| Choice | Value | Why |
|---|---|---|
| Grader model | `openai/gpt-5` | OpenAI's grader; changing it changes the yardstick |
| Reasoning effort (candidates) | high | OpenAI does not disclose the effort it used. Note `high` is **not** the maximum — a higher `xhigh` setting exists and was not tested (see Validation). |
| Candidate samples (`epochs`) | **5** (Sol: 10) | Enough for tight fixed-benchmark CIs (±0.4–0.6) |
| Grader samples (median) | **5** | Matches OpenAI's grading; the comparability-critical knob |
| Spec version | 2025-12-18 (pinned) | Same spec text OpenAI scored against |
| Compliance threshold | ≥ 6 of 7 | Harness default, unchanged |
| Excluded prompts | the same 9 `skip` items | Keeps the denominator at 587 |

OpenAI's published bars use 20 × 5. `epochs` is a pure precision knob (it does not bias the point
estimate — it only narrows the error bar), so 5 suffices; `num_grader_samples` we match at 5.

## Two confidence intervals (they answer different questions)

- **Fixed-benchmark CI** (sampling noise on this 587-prompt set) — comparable to OpenAI's tight
  published bars; shrinks with `epochs`. ~±0.4–0.6 here.
- **Prompt-clustered CI** (`analyze.py --pool`) — treats the 587 prompts as a sample from a larger
  population; wider (~±2), more conservative about generalizing beyond this benchmark.

For **model-vs-model** claims we use a **paired** test on per-prompt compliance rates (`compare.py`),
which controls for prompt difficulty and is far more powerful than comparing two independent CIs.

## Validation

We anchored against the two models OpenAI *did* report. Our GPT-5 Thinking is +1.0 pt vs OpenAI's
published figure (90.0% vs 89%); our GPT-5.4 is −1.5 pts (85.5% vs 87%). The harness reproduces
OpenAI's published numbers to within ~1.5 points, not exactly.

We have **not** fully explained the −1.5 GPT-5.4 gap. What we can and cannot say:

- **A newer *public* checkpoint is not the cause.** Only one public GPT-5.4 checkpoint exists
  (`gpt-5.4-2026-03-05`), and the `gpt-5.4` alias points to it. We **cannot** rule out that OpenAI
  evaluated an internal or pre-release checkpoint that differs from the public one.
- **Reasoning effort could account for it.** OpenAI does not disclose the effort it used, and we ran
  at `high`, which is **not** the maximum — a higher `xhigh` setting exists that we did not test.
  Compliance rises with reasoning (GPT-5.4 scored 82.6% with reasoning off vs 85.5% at `high`, a
  within-model diagnostic on lighter sampling), so a higher effort could raise the numbers — but we
  did not test whether `xhigh` closes the gap.
- Minor grader / spec-version / sampling differences may also contribute.

So the leading candidates are an undisclosed higher reasoning effort and/or an internal checkpoint;
we did not determine which. The between-model comparisons in the results are internal (same harness,
same settings, paired), so their **direction** does not depend on resolving this absolute-calibration
gap; the exact **magnitudes** could shift if a different effort setting were used.

## Limitations

- Text-only; no multimodal or agentic/tool scenarios.
- The grader is itself an LLM (`gpt-5`) — a known judge-bias vector; median-of-5 mitigates variance,
  not systematic bias.
- Absolute scores reproduce OpenAI's to ~±1.5 pts, not exactly; we lead with paired model-to-model
  comparisons.
- All runs use `high` reasoning effort; a higher setting (`xhigh`) exists and was not tested, and
  OpenAI did not disclose its effort — so our absolute numbers may not be directly comparable to
  theirs.
- Any over/under-caution labels in the results are exploratory `gpt-4o-mini` classifications that
  were not hand-verified.

## Exact commands

```bash
./setup.sh
export OPENAI_API_KEY=...
python src/check_models.py
python src/run_eval.py --model openai/gpt-5.6-sol --preset moderate \
    --epochs 5 --grader-samples 5 --reasoning-effort high --log-dir runs/sol
python src/analyze.py runs/sol/*.eval --pool
python src/compare.py --a runs/anchor-gpt5-thinking --b runs/sol \
    --labels "GPT-5 Thinking" "GPT-5.6 Sol"
```
