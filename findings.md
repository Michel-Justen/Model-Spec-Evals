# Preliminary findings

An early, underpowered pilot — 50 prompts out of 587, one sample each. Treat these as
directional, not final.

## Setup

50 prompts, 10 drawn from each of the 5 spec sections. Each model answered once and was
graded once by GPT-5. Candidates run at high reasoning effort.

## Result

| Model | Adherence |
|---|---|
| GPT-5.4 | 92% |
| GPT-5.5 | 90% |
| GPT-5.6 Sol | 88% |

(chart in `Adherence.png`)

At 50 prompts the 95% confidence interval is about ±8 points, so the 92 → 90 → 88 slope is
within noise — I can't yet say GPT-5.6 Sol is actually less adherent than GPT-5.4. What I can
say is there's no sign of improvement across the three, and maybe a slight drift downward,
which is worth settling with a full run.

As a sanity check on the setup: GPT-5.4 scored 92% here vs OpenAI's published ~87% (on the
full 587 prompts, 20 samples). Same ballpark, which suggests the harness is faithful — the
gap is what you'd expect from the smaller, non-representative subset.

## Caveats

- Text only.
- The grader is itself an LLM (GPT-5), so grading carries some built-in judge bias.
- Absolute scores aren't perfectly reproducible; the model-to-model comparison is the point.
