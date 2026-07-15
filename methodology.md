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
| Reasoning effort (candidates) | **high** | Best-case / charitable; compliance is effort-dependent (§Validation) |
| Candidate samples (`epochs`) | **5** (Sol: 10) | Enough for tight fixed-benchmark CIs (±0.4–0.6) |
| Grader samples (median) | **5** | Matches OpenAI's grading; the comparability-critical knob |
| Spec version | 2025-12-18 (pinned) | Same spec text OpenAI scored against |
| Compliance threshold | ≥ 6 of 7 | Harness default, unchanged |
| Excluded prompts | the same 9 `skip` items | Keeps the denominator at 587 |

OpenAI's published bars use 20 × 5. `epochs` is a pure precision knob (it does not bias the point
estimate — it only narrows the error bar), so 5 suffices; `num_grader_samples` we match at 5.

## Two confidence intervals (they answer different questions)

- **Fixed-benchmark CI** (sampling noise on this 587-prompt set) — comparable to OpenAI's tight
  published bars; shrinks with `epochs`. Used on the charts. ~±0.4–0.6 here.
- **Prompt-clustered CI** (`analyze.py --pool`) — treats the 587 prompts as a sample from a larger
  population; wider (~±2), more conservative about generalizing beyond this benchmark.

For **model-vs-model** claims we use a **paired** test on per-prompt compliance rates (`compare.py`),
which controls for prompt difficulty and is far more powerful than comparing two independent CIs.

## Validation

We anchored against the two models OpenAI *did* report. GPT-5 Thinking reproduces well (90.0% vs
89%); GPT-5.4 is 1.5 pts low (85.5% vs 87%). The anchors miss in opposite directions, so this is
**reproduction scatter of ~±1.5 pts, not a systematic bias** — the harness reproduces OpenAI's
published figures to within ~1.5 pts, not exactly. Two diagnostics on the 5.4 gap:

- **Version drift — ruled out.** Only one GPT-5.4 checkpoint exists; the alias points to it.
- **Reasoning effort — not the cause, but it matters.** GPT-5.4 at *no* reasoning = 82.6% vs 85.5%
  at high, so more reasoning → more compliance and we already use the max. See [RESULTS.md §3](RESULTS.md#3-validation--read-this-before-trusting-the-absolute-numbers).

Because the GPT-5.5 / Sol comparisons are internal, same-methodology, and paired against GPT-5
Thinking (which we reproduce), their **direction and significance are robust** to this scatter.

## Limitations

- Text-only; no multimodal or agentic/tool scenarios.
- The grader is itself an LLM (`gpt-5`) — a known judge-bias vector; median-of-5 mitigates variance,
  not systematic bias.
- Absolute scores reproduce OpenAI's to ~±1.5 pts, not exactly; we lead with paired model-to-model
  comparisons.

## Exact commands

```bash
./setup.sh
export OPENAI_API_KEY=...
python src/check_models.py
python src/run_eval.py --model openai/gpt-5.6-sol --preset moderate \
    --epochs 5 --grader-samples 5 --reasoning-effort high --log-dir runs/sol
python src/analyze.py runs/sol/*.eval --pool
python src/compare.py --a runs/anchor-gpt5-thinking runs/anchor-topup --b runs/sol runs/sol-topup \
    --labels "GPT-5 Thinking" "GPT-5.6 Sol"
```
